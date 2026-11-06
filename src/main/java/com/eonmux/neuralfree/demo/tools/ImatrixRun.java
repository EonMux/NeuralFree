package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.ImatrixCollector;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuMatmulTap;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.ops.flow.NextnDraftModel;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * One activation collection of {@code nftool imatrix} over an open engine model (llama.cpp
 * {@code compute_imatrix}): the model on the CPU engine ({@code -ngl 0}) or the GPU engine, an
 * {@link ImatrixCollector} fed by the CPU matmul tap ({@link CpuMatVec#observeGlobally}) and, on the GPU engine,
 * by the device matmul tap ({@link GpuMatmulTap}), and, under {@code --nextn} or {@code --model-draft}, the NextN/MTP
 * head replayed over the trunk's hidden rows after every decode ({@link NextnDraftModel#replay}).
 *
 * <p>The corpus runs in groups of {@code n_seq = max(1, n_batch / n_ctx)} whole-context chunks: each group clears
 * the memory, then decodes its chunks as parallel sequences in batches of {@code n_batch} tokens (several batches
 * per chunk when {@code n_batch < n_ctx}), the first token of every chunk replaced by BOS when the vocabulary adds
 * one. Several sequences run as one multi-sequence {@link ForwardBatch} over paged memory of {@code n_seq}
 * sequences; one sequence runs the contiguous single-sequence calls. The chunk counter advances by the group's
 * chunk count and the output and save frequencies fire when the counter crosses a multiple of them, as the
 * reference's collector saves.</p>
 */
final class ImatrixRun implements AutoCloseable {

    /** Saves the matrix with the given chunk count; {@code atChunk >= 0} writes the {@code .at_<n>} copy. */
    interface Checkpoint {
        void save(List<ImatrixFile.Entry> collected, int chunkCount, int atChunk) throws IOException;
    }

    private final ImatrixCommand.Params params;
    private final EngineModel model;
    private final int nSeq;
    private final int nBatch;
    private final ImatrixCollector collector;
    private SpeculativeSession session;
    private NextnDraftModel nextn;
    private CpuWeights hostWeights;

    private ImatrixRun(ImatrixCommand.Params params, EngineModel model, int nSeq) {
        this.params = params;
        this.model = model;
        this.nSeq = nSeq;
        this.nBatch = Math.min(params.batchSize, nSeq * params.ctxSize);
        this.collector = new ImatrixCollector(this::nameHostWeight,
                name -> ImatrixCommand.collectable(name, params.processOutput, params.nextn),
                ImatrixCollector.LLAMA_MIN_DENSE_ROWS);
    }

    /**
     * Opens the model of {@code params} for the collection: the memory holds {@code n_seq} sequences of
     * {@code --ctx-size} tokens, and the NextN head is bound when the run collects it.
     *
     * @throws ImatrixCommand.UsageException when the flags do not fit the model (no token loop, NextN with
     *                                       several sequences, an incompatible draft), naming the flag
     */
    static ImatrixRun open(ImatrixCommand.Params params) throws IOException, ImatrixCommand.UsageException {
        int nSeq = sequences(params);
        MemoryConfig.Builder memory = MemoryConfig.builder().nCtx(Math.multiplyExact(nSeq, params.ctxSize));
        if (nSeq > 1) {
            memory.parallel(nSeq);
        }
        EngineConfig cfg = params.placed(EngineConfig.builder()
                .backend(params.nGpuLayers == 0 ? EngineConfig.BackendMode.CPU : EngineConfig.BackendMode.AUTO)
                .gpuLayers(params.nGpuLayers == 0 ? EngineConfig.GPU_LAYERS_AUTO : params.nGpuLayers)
                .memory(memory.build())
                .build());
        EngineModel m = LLMModelFactory.open(params.model.toString(), cfg);
        ImatrixRun run = new ImatrixRun(params, m, nSeq);
        try {
            run.check();
            if (params.nextn) {
                run.bindNextn(cfg);
            }
            return run;
        } catch (IOException | ImatrixCommand.UsageException | RuntimeException e) {
            run.close();
            throw e;
        }
    }

    /**
     * {@code n_seq} of llama.cpp's imatrix main: {@code max(1, n_batch / n_ctx)} sequences, a single one under
     * {@code --nextn}/{@code --model-draft}.
     */
    static int sequences(ImatrixCommand.Params params) throws ImatrixCommand.UsageException {
        int ctx = params.ctxSize;
        int nSeq = Math.max(1, params.batchSize / ctx);
        if (params.batchSize >= ctx && params.batchSize % ctx != 0) {
            throw new ImatrixCommand.UsageException("--batch-size " + params.batchSize + " must be below --ctx-size "
                    + ctx + " or a multiple of it");
        }
        if (params.nextn && nSeq > 1) {
            throw new ImatrixCommand.UsageException("'--nextn' and '--model-draft' need a single sequence per batch,"
                    + " set '--batch-size' to at most '--ctx-size' (" + ctx + ")");
        }
        if (nSeq > MemoryConfig.MAX_PARALLEL) {
            throw new ImatrixCommand.UsageException("--batch-size " + params.batchSize + " runs " + nSeq
                    + " sequences of " + ctx + " tokens, more than the " + MemoryConfig.MAX_PARALLEL
                    + " the memory holds");
        }
        return nSeq;
    }

    private void check() throws ImatrixCommand.UsageException {
        if (params.ctxSize > model.contextLength()) {
            System.err.println("nftool imatrix: warning: the model was trained on only " + model.contextLength()
                    + " context tokens (" + params.ctxSize + " requested)");
        }
        SupportClass support = model.supportClass();
        if (model.forwardBackend().memory() == null || !hasTokenLoop(support)) {
            throw new ImatrixCommand.UsageException("the collector runs the decoder matmuls of a model with memory; "
                    + model.architecture() + " (" + support + ") has none");
        }
        if (!(model instanceof GGUFModel) && !(model instanceof GGUFModelGPU)) {
            throw new ImatrixCommand.UsageException("the imatrix collection runs the GGUF engines, but "
                    + params.model + " opened as " + model.getClass().getSimpleName());
        }
        PagedSequences paged = model.forwardBackend().memory().pagedSequences();
        if (nSeq > 1 && (paged == null || paged.maxSequences() < nSeq)) {
            throw new ImatrixCommand.UsageException("--batch-size " + params.batchSize + " runs " + nSeq
                    + " sequences, but the memory of " + model.architecture() + " holds "
                    + (paged == null ? "one" : paged.maxSequences()) + "; set --batch-size to at most --ctx-size");
        }
    }

    /**
     * Binds the NextN head (llama.cpp's {@code nextn_collector}): the model's own NextN blocks under
     * {@code --nextn}, the MTP draft file under {@code --model-draft}; a model without NextN blocks runs without
     * one, as the reference warns.
     */
    private void bindNextn(EngineConfig cfg) throws IOException, ImatrixCommand.UsageException {
        boolean own = params.modelDraft == null;
        String flag = own ? "'--nextn'" : "'--model-draft'";
        if (!own && !model.spec().nextnLayers().isEmpty()) {
            throw new ImatrixCommand.UsageException("--model-draft: the model already includes NextN layers");
        }
        if (own && model.spec().nextnLayers().isEmpty()) {
            System.err.println("nftool imatrix: warning: the model has no NextN layers, " + flag + " has no effect");
            return;
        }
        EngineConfig drafter = own ? cfg.toBuilder().mtp(true).build()
                : cfg.toBuilder().draftModelPath(params.modelDraft.toString())
                        .draftType(EngineConfig.DraftType.MTP).build();
        try {
            session = LLMModelFactory.speculate(model, drafter);
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
            throw new ImatrixCommand.UsageException(flag + ": " + e.getMessage());
        }
        if (!(session.drafter() instanceof NextnDraftModel)) {
            throw new ImatrixCommand.UsageException(flag + ": the drafter is " + session.drafterKind()
                    + ", not a NextN/MTP head");
        }
        nextn = (NextnDraftModel) session.drafter();
        if (nextn.nextnLayers() > 1) {
            throw new ImatrixCommand.UsageException("multi-layer NextN drafts are not supported (found "
                    + nextn.nextnLayers() + ")");
        }
        System.out.println("nftool imatrix: processing " + nextn.nextnLayers() + " NextN layer(s) from block "
                + model.spec().nLayer());
    }

    /** {@code imatrix.n_layer_nextn} of the run: the NextN blocks the head replays, 0 without one. */
    int nLayerNextn() {
        return nextn == null ? 0 : nextn.nextnLayers();
    }

    /** The sequences each decode group runs. */
    int sequences() {
        return nSeq;
    }

    /** The model the run collects over. */
    EngineModel model() {
        return model;
    }

    /** The statistics collected so far. */
    List<ImatrixFile.Entry> entries() {
        return collector.entries();
    }

    /**
     * Runs the corpus windows and saves at the frequencies.
     *
     * @param window {@code {startToken, nChunk}} of {@link ImatrixCommand#chunkWindow}
     * @param seed   the chunk counter of the combined {@code --in-file} matrices
     * @return the final chunk count
     */
    int run(int[] tokens, int[] window, int seed, ImatrixPerplexity ppl, Checkpoint checkpoint) throws IOException {
        CpuMatVec.observeGlobally(collector);
        GpuMatmulTap.observeGlobally(model instanceof GGUFModelGPU ? new GpuMatmulTap.Observer() {
            @Override
            public boolean wants(String weight, boolean indirect, int rows) {
                return collector.wants(weight, indirect, rows);
            }

            @Override
            public void observe(String weight, int matrices, int matrix, float[] x, int xOff, int xStride, int cols,
                                int nTokens) {
                collector.accumulate(weight, matrices, matrix, x, xOff, xStride, cols, nTokens);
            }
        } : null);
        try {
            return chunks(tokens, window, seed, ppl, checkpoint);
        } finally {
            CpuMatVec.observeGlobally(null);
            GpuMatmulTap.observeGlobally(null);
        }
    }

    private int chunks(int[] tokens, int[] window, int seed, ImatrixPerplexity ppl, Checkpoint checkpoint)
            throws IOException {
        int nChunk = window[1];
        System.out.println("nftool imatrix: computing over " + nChunk + " chunks, n_ctx=" + params.ctxSize
                + ", batch_size=" + nBatch + ", n_seq=" + nSeq + " on the "
                + (model instanceof GGUFModelGPU ? "GPU" : "CPU") + " engine");
        int lastChunk = seed;
        for (int c = 0; c < nChunk; c += nSeq) {
            int nSeqBatch = Math.min(nSeq, nChunk - c);
            group(tokens, window[0] + c * params.ctxSize, c, nSeqBatch, ppl);
            lastChunk += nSeqBatch;
            List<ImatrixFile.Entry> collected = null;
            if ((lastChunk % params.outputFrequency) / nSeqBatch == 0) {
                collected = collector.entries();
                checkpoint.save(collected, lastChunk, -1);
            }
            if (params.saveFrequency > 0 && (lastChunk % params.saveFrequency) / nSeqBatch == 0) {
                checkpoint.save(collected != null ? collected : collector.entries(), lastChunk, lastChunk);
            }
        }
        if (params.ppl) {
            System.out.println();
        }
        return lastChunk;
    }

    /**
     * One group of {@code nSeqBatch} chunks from token {@code start}: the memory cleared, the batches decoded, the
     * NextN head replayed after each, and the perplexity over the second half of every chunk.
     */
    private void group(int[] tokens, int start, int firstChunk, int nSeqBatch, ImatrixPerplexity ppl)
            throws IOException {
        ForwardBackend fb = model.forwardBackend();
        fb.reset();
        if (nextn != null) {
            nextn.reset();
        }
        int ctx = params.ctxSize;
        float[][][] logits = params.ppl ? new float[nSeqBatch][ctx][] : null;
        for (int b0 = 0; b0 < ctx; b0 += nBatch) {
            int size = Math.min(ctx - b0, nBatch);
            float[][] rows = decode(tokens, start, nSeqBatch, b0, size);
            if (nextn != null) {
                nextn.replay();
            }
            if (logits != null) {
                for (int s = 0; s < nSeqBatch; s++) {
                    for (int k = 0; k < size; k++) {
                        logits[s][b0 + k] = rows[s * size + k].clone();
                    }
                }
            }
        }
        if (logits == null) {
            return;
        }
        for (int s = 0; s < nSeqBatch; s++) {
            int seqStart = start + s * ctx;
            for (int r = ctx / 2; r < ctx - 1; r++) {
                ppl.observe(logits[s][r], tokens[seqStart + r + 1]);
            }
            System.out.printf(Locale.ROOT, "[%d]%.4f,", firstChunk + s + 1, ppl.runningPpl());
        }
    }

    /**
     * Decodes positions {@code [b0, b0 + size)} of {@code nSeqBatch} chunks: one sequence through the contiguous
     * calls, several as one multi-sequence batch. Returns every row's logits, sequence-major, when the run needs
     * them (perplexity, the head collection), else null.
     */
    private float[][] decode(int[] tokens, int start, int nSeqBatch, int b0, int size) throws IOException {
        boolean allRows = params.ppl || params.processOutput;
        ForwardBackend fb = model.forwardBackend();
        try {
            if (nSeq == 1) {
                int[] slice = slice(tokens, start, b0, size);
                if (allRows) {
                    return fb.forwardRows(slice, b0, null);
                }
                fb.prefill(slice, b0);
                return null;
            }
            ForwardBatch.Builder batch = ForwardBatch.builder(nSeqBatch * size);
            for (int s = 0; s < nSeqBatch; s++) {
                int[] slice = slice(tokens, start + s * params.ctxSize, b0, size);
                for (int k = 0; k < size; k++) {
                    batch.add(slice[k], b0 + k, s, allRows || k == size - 1);
                }
            }
            float[][] out = fb.forwardBatch(batch.build());
            return allRows ? out : null;
        } catch (ContextFullException e) {
            throw new IOException("the chunk at token " + start + " does not fit the " + params.ctxSize
                    + "-token window: " + e.getMessage(), e);
        }
    }

    /** Tokens {@code [b0, b0 + size)} of the chunk at {@code chunkStart}; its first token is BOS when added. */
    private int[] slice(int[] tokens, int chunkStart, int b0, int size) {
        int[] out = Arrays.copyOfRange(tokens, chunkStart + b0, chunkStart + b0 + size);
        GGUFTokenizer tok = model.tokenizer();
        if (b0 == 0 && tok.vocab().addBos() && tok.bosId() >= 0) {
            out[0] = tok.bosId();
        }
        return out;
    }

    /**
     * The stored GGUF name of a CPU weight: the CPU engine's catalog, the GPU engine's host weights (layers it
     * keeps on the CPU), or the NextN head's draft file.
     *
     * @throws IllegalStateException when no catalog holds the weight
     */
    private String nameHostWeight(CpuTensorRef w) {
        if (model instanceof GGUFModel) {
            CpuWeights weights = ((GGUFModel) model).forwardContext().weights();
            if (weights.owns(w)) {
                return weights.describe(w).name();
            }
        } else if (model instanceof GGUFModelGPU) {
            CpuWeights weights = gpuHostWeights();
            if (weights.owns(w)) {
                return weights.describe(w).name();
            }
        }
        String drafter = nextn == null ? null : nextn.hostWeightName(w);
        if (drafter == null) {
            throw new IllegalStateException("the imatrix collector cannot name the host weight " + w);
        }
        return drafter;
    }

    private CpuWeights gpuHostWeights() {
        if (hostWeights == null) {
            try {
                hostWeights = ((GGUFModelGPU) model).cpuWeights();
            } catch (IOException e) {
                throw new IllegalStateException("the host weights of " + params.model + " cannot be opened", e);
            }
        }
        return hostWeights;
    }

    /** Whether models of {@code support} run a decoder-only token loop (LLMModelFactory's set). */
    private static boolean hasTokenLoop(SupportClass support) {
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
            case DRAFT:
                return false;
            default:
                return true;
        }
    }

    @Override
    public void close() throws IOException {
        try {
            if (session != null) {
                session.close();
            }
        } finally {
            model.close();
        }
    }
}
