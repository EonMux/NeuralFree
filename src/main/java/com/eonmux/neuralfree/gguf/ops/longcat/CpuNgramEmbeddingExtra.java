package com.eonmux.neuralfree.gguf.ops.longcat;

import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.Objects;

/**
 * LongCat-Flash-Lite's fused n-gram input embedding, replacing the token embedding on the first block (vLLM
 * longcat_flash_ngram.py {@code NgramEmbedding.embed_batched}): the token row is averaged with one projected hashed
 * row per embedder, {@code (word + sum_e proj_e(table_e[id_e])) / (embedders + 1)}.
 *
 * <p>The hash is integer bookkeeping over the step's token record ({@link CpuScratch#TOKEN_IDS}) and the memory's
 * per-sequence {@link SequenceState}, computed by {@link NgramHash} exactly as vLLM's kernel does. The tables are one
 * concatenated matrix, so the per-embedder gathers are one matmul against the stacked projections. An embedding row
 * is refused by the model input ({@code EmbeddingRowSupport}) since its history has no token record.</p>
 *
 * <p>The extra writes the residual stream of the first block, whose width must be {@code n_embd}. Not thread-safe.</p>
 */
public final class CpuNgramEmbeddingExtra implements CpuLayerExtra {

    private static final String T_IDS = "longcat.ngram.ids";
    private static final String T_ROWS = "longcat.ngram.rows";
    private static final String T_PROJ = "longcat.ngram.proj";

    private CpuWeights weights;
    private CpuMatVec matVec;
    private CpuScratch scratch;
    private CpuMemory memory;
    private CpuTensorRef table;
    private CpuTensorRef projection;
    private NgramHash hash;
    private int[] ids;
    private float[] rows;
    private float[] projected;
    private int ubatch;
    private int nEmbd;
    private int width;
    private int embedders;

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (hash != null) {
            throw new IllegalStateException("LongCat n-gram embedding extra is already bound");
        }
        EmbeddingSpec embd = ctx.spec().embedding();
        String where = ctx.spec().arch().ggufName() + " layer " + layer.stack() + "." + layer.il();
        if (!embd.ngram()) {
            throw new IllegalArgumentException(where + " lists '" + LayerSpec.Extras.NGRAM_EMBD
                    + "' but the model input is not an n-gram embedding");
        }
        if (!ctx.hasMemory()) {
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
        this.weights = ctx.weights();
        this.matVec = ctx.matVec();
        this.scratch = ctx.scratch();
        this.memory = ctx.memory();
        this.table = weights.get(embd.ngramEmbd());
        this.projection = weights.get(embd.ngramProj());
        if (table.cols() != width) {
            throw new IllegalArgumentException(where + ": " + table + " rows hold " + table.cols()
                    + " values, expected n_embd / embedders = " + width);
        }
        long expected = NgramHash.expectedRows(embd.ngramVocabSize(), embd.ngramSplits(), embd.ngramNeighbors());
        if (table.rows() != expected) {
            throw new IllegalArgumentException(where + ": " + table + " holds " + table.rows()
                    + " rows, expected " + expected + " for the declared n-gram geometry");
        }
        if (projection.cols() != (long) embedders * width || projection.rows() != nEmbd) {
            throw new IllegalArgumentException(where + ": " + projection + " maps " + projection.cols()
                    + " -> " + projection.rows() + " values, expected " + (embedders * width) + " -> " + nEmbd);
        }
        this.ids = scratch.intTemp(T_IDS, Math.multiplyExact(ubatch, embedders));
        this.rows = scratch.temp(T_ROWS, Math.multiplyExact(Math.multiplyExact(ubatch, embedders), width));
        this.projected = scratch.temp(T_PROJ, Math.multiplyExact(ubatch, nEmbd));
    }

    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        checkStep(n, x);
        run(scratch.ints(CpuScratch.TOKEN_IDS), 0, n, memory.sequence(), true, pos0, 0, x);
    }

    /**
     * One batched step: every segment hashes against its own sequence's record and rewrites its own rows of the
     * residual stream.
     */
    @Override
    public void beforeBlock(StepLayout s, float[] x) {
        checkStep(s.nTokens(), x);
        int[] step = scratch.ints(CpuScratch.TOKEN_IDS);
        boolean paged = memory.pagedSequences() != null;
        for (int seg = 0; seg < s.nSeg(); seg++) {
            SequenceState record = paged ? memory.pagedSequences().sequence(s.seq(seg)) : memory.sequence();
            run(step, s.tokStart(seg), s.len(seg), record, !paged, s.pos0(seg), s.tokStart(seg), x);
        }
    }

    /** Rewrites rows {@code [row0, row0 + n)} of the stream with the fused embedding of their tokens. */
    private void run(int[] step, int tokStart, int n, SequenceState record, boolean contiguous, int pos0, int row0,
                     float[] x) {
        if (n == 0) {
            return;
        }
        hash.ids(step, tokStart, n, record, contiguous, pos0, ids);
        int gathered = n * embedders;
        for (int i = 0; i < gathered; i++) {
            weights.embedRow(table, ids[i], rows, i * width);
        }
        matVec.matmul(projection, rows, 0, projected, 0, n);
        float scale = 1.0f / (embedders + 1);
        for (int t = 0; t < n; t++) {
            int dst = (row0 + t) * nEmbd;
            int src = t * nEmbd;
            for (int i = 0; i < nEmbd; i++) {
                x[dst + i] = (x[dst + i] + projected[src + i]) * scale;
            }
        }
    }

    private void checkStep(int n, float[] x) {
        if (hash == null) {
            throw new IllegalStateException("LongCat n-gram embedding extra is not bound");
        }
        if (n < 0 || n > ubatch) {
            throw new IllegalArgumentException("n-gram step of " + n + " rows is outside [1, " + ubatch + "]");
        }
        if (x.length < (long) n * nEmbd) {
            throw new IllegalArgumentException("residual stream holds " + x.length + " floats, the n-gram embedding"
                    + " needs " + ((long) n * nEmbd));
        }
    }

    @Override
    public String toString() {
        return "CpuNgramEmbeddingExtra{embedders=" + embedders + ", width=" + width + '}';
    }
}
