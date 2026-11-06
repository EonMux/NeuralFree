package com.eonmux.neuralfree.gguf.ops.longcat;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * The GPU twin of {@link CpuNgramEmbeddingExtra}: the n-gram hash runs on the host over the step's token record
 * ({@link GpuScratch#readStepTokens}), the gathers are one {@code GpuOps.embed}, and the projection and the
 * weighted average with the token embedding are shared kernels.
 *
 * <p>The GPU holds the same concatenated table and stacked projection as the CPU: the host ids are laid out
 * token-major, so one lookup writes the {@code embedders} rows of a token consecutively, the width of one row of the
 * projection's input. Not thread-safe.</p>
 */
public final class GpuNgramEmbeddingExtra implements GpuLayerExtra {

    private static final String T_IDS = "longcat.ngram.ids";
    private static final String T_ROWS = "longcat.ngram.rows";
    private static final String T_PROJ = "longcat.ngram.proj";

    private GpuForwardContext ctx;
    private GpuOps ops;
    private GpuScratch scratch;
    private GpuMemory memory;
    private GpuWeight table;
    private GpuWeight projection;
    private NgramHash hash;
    private ScratchSlot rows;
    private ScratchSlot projected;
    private int[] stepIds;
    private int[] ids;
    private int ubatch;
    private int nEmbd;
    private int width;
    private int embedders;

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (this.ctx != null) {
            throw new IllegalStateException("LongCat n-gram embedding extra is already bound");
        }
        EmbeddingSpec embd = ctx.spec().embedding();
        String where = ctx.spec().arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
        if (!embd.ngram()) {
            throw new IllegalArgumentException(where + " lists '" + LayerSpec.Extras.NGRAM_EMBD
                    + "' but the model input is not an n-gram embedding");
        }
        GpuMemory mem = ctx.memory();
        if (mem == null) {
            throw new IllegalStateException(where + ": the n-gram hash needs the model memory's token record");
        }
        this.nEmbd = embd.nEmbd();
        if (layer.residual().streamWidth() != nEmbd) {
            throw new IllegalArgumentException(where + ": the n-gram embedding needs an n_embd-wide stream, got "
                    + layer.residual().streamWidth());
        }
        this.ubatch = ctx.scratch().ubatch();
        this.hash = new NgramHash(embd.ngramVocabSize(), embd.ngramSplits(), embd.ngramNeighbors(),
                embd.ngramEosTokenId(), ctx.spec().hparams().nVocab(), ubatch);
        this.embedders = hash.embedders();
        this.width = nEmbd / embedders;
        if (nEmbd % embedders != 0 || width < 1) {
            throw new IllegalArgumentException(where + ": n_embd " + nEmbd + " is not divisible by the "
                    + embedders + " n-gram embedders");
        }
        this.ctx = ctx;
        this.ops = ctx.ops();
        this.scratch = ctx.scratch();
        this.memory = mem;
        GpuWeights weights = ctx.weights();
        this.table = weights.get(embd.ngramEmbd().name());
        this.projection = weights.get(embd.ngramProj().name());
        if (table.cols() != width) {
            throw new IllegalArgumentException(where + ": " + table.name() + " rows hold " + table.cols()
                    + " values, expected n_embd / embedders = " + width);
        }
        long expected = NgramHash.expectedRows(embd.ngramVocabSize(), embd.ngramSplits(), embd.ngramNeighbors());
        if (table.rows() != expected) {
            throw new IllegalArgumentException(where + ": " + table.name() + " holds " + table.rows()
                    + " rows, expected " + expected + " for the declared n-gram geometry");
        }
        if (projection.cols() != (long) embedders * width || projection.rows() != nEmbd) {
            throw new IllegalArgumentException(where + ": " + projection.name() + " maps " + projection.cols()
                    + " -> " + projection.rows() + " values, expected " + (embedders * width) + " -> " + nEmbd);
        }
        this.rows = scratch.temp(T_ROWS, (long) ubatch * embedders * width);
        this.projected = scratch.temp(T_PROJ, (long) ubatch * nEmbd);
        this.stepIds = new int[ubatch];
        this.ids = new int[Math.multiplyExact(ubatch, embedders)];
    }

    @Override
    public void beforeBlock(int pos0, int n, ScratchSlot x) {
        checkStep(n, x);
        scratch.readStepTokens(stepIds, 0, n);
        run(0, n, memory.sequence(), true, pos0, x);
    }

    @Override
    public void beforeBlock(StepLayout s, ScratchSlot x) {
        checkStep(s.nTokens(), x);
        scratch.readStepTokens(stepIds, 0, s.nTokens());
        boolean paged = memory.pagedSequences() != null;
        for (int seg = 0; seg < s.nSeg(); seg++) {
            SequenceState record = paged ? memory.pagedSequences().sequence(s.seq(seg)) : memory.sequence();
            run(s.tokStart(seg), s.len(seg), record, !paged, s.pos0(seg), x);
        }
    }

    /** Rewrites the segment's {@code n} rows of {@code x} with the fused embedding of their tokens. */
    private void run(int tokStart, int n, SequenceState record, boolean contiguous, int pos0, ScratchSlot x) {
        if (n == 0) {
            return;
        }
        hash.ids(stepIds, tokStart, n, record, contiguous, pos0, ids);
        int gathered = Math.multiplyExact(n, embedders);
        ops.embed(table, ids, gathered, rows.buffer(), rows.offset());
        ops.matmul(projection, rows.buffer(), rows.offset(), projected.buffer(), projected.offset(), n);
        float scale = 1.0f / (embedders + 1);
        int count = Math.multiplyExact(n, nEmbd);
        int xOff = x.offset() + Math.multiplyExact(tokStart, nEmbd);
        x.require((long) xOff + count);
        ops.scale(x.buffer(), xOff, count, scale);
        ops.scaleAdd(projected.buffer(), projected.offset(), scale, x.buffer(), xOff, count);
    }

    private void checkStep(int n, ScratchSlot x) {
        if (ctx == null) {
            throw new IllegalStateException("LongCat n-gram embedding extra is not bound");
        }
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("n-gram step of " + n + " rows is outside [1, " + ubatch + "]");
        }
        x.require((long) n * nEmbd);
    }

    @Override
    public String toString() {
        return "GpuNgramEmbeddingExtra{embedders=" + embedders + ", width=" + width + '}';
    }
}
