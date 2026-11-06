package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Embedding stage of the GPU forward path ({@link EmbeddingSpec}): writes the residual stream
 * {@link ScratchSlots#X} for the tokens of one ubatch. Per step, in llama.cpp order:
 * <ol>
 *   <li>token ids (and token types when the model has a type table) are written to {@link ScratchSlots#TOKEN_IDS}
 *       in one transfer, laid out {@code [ids][types]}, and the ids are recorded on the host
 *       ({@link GpuScratch#recordStepTokens});</li>
 *   <li>{@code x = token_embd[id]} through {@link GpuOps#embed} (any table type, row shards, inverse Hadamard);</li>
 *   <li>{@code x += scale * B A[id]}: the LoRA deltas registered on {@code token_embd.weight}
 *       ({@link GpuProjectionDeltas#serveEmbedding}, {@code build_inp_embd} llama-graph.cpp:2694-2709);</li>
 *   <li>{@code x *= embdScale}: {@code build_inp_embd} scales the token rows before the graph adds anything
 *       (llama-graph.cpp:2759-2780);</li>
 *   <li>{@code x = project_in · x} when {@link EmbeddingSpec#inputProjection()} maps a token table narrower than
 *       {@code n_embd} (OPT's {@code word_embed_proj_dim});</li>
 *   <li>{@code x += token_types[type]} (models/bert.cpp:90-94; a null type array selects row 0 as llama.cpp
 *       does) and {@code x += position_embd[pos + offset]} (bert.cpp:95-97, gpt2.cpp,
 *       {@link EmbeddingSpec#posOffset()});</li>
 *   <li>{@code token_embd_norm}: LayerNorm with its bias (bert.cpp:101, bloom.cpp:76-81, rwkv6.cpp:102), or the
 *       RMS input norm with {@code f_norm_rms_eps} ({@link EmbeddingSpec#inputRmsNorm()}, talkie.cpp:50-51,
 *       muse-spark.cpp:86-87), weighted when {@link EmbeddingSpec#inputRmsNormWeight()} is set (Inkling
 *       {@code embed_norm});</li>
 *   <li>hyper-connection models repeat each row over the {@code hc} streams of the first layer
 *       (deepseek4.cpp:1240-1242).</li>
 * </ol>
 * {@link ScratchSlots#INPUT_EMBD} receives the stage output as {@code [t][nEmbdInp]} rows when the model reads it
 * again: zero-padded past {@code n_embd} for deepstack models (llama-graph.cpp {@code ggml_pad}, then a view of the
 * first {@code n_embd} columns as the stream, so the deepstack slices of token rows are zero) and as the source of
 * the {@code embd_skip} layer extra.
 *
 * <p>Embedding rows ({@code prompt_embeds}, projected media rows) replace the token-table lookup of their rows, as
 * the CPU stage does: the row's first {@code n_embd} values take the place of steps 2-4 (the lookup, its LoRA deltas
 * and the embedding scale belong to the lookup), the type and position additions and the input norm run as for
 * tokens, and a row of the full input width {@code nEmbdInp} keeps its tail in {@link ScratchSlots#INPUT_EMBD} (an
 * {@code n_embd} row is zero-padded). A step may mix token and embedding rows; the step's host token record
 * ({@link GpuScratch#recordStepTokens}) holds the null token {@code -1} for an embedding row.</p>
 *
 * <p>Per step the stage allocates no device memory and never waits on the device. Not thread-safe.</p>
 */
public final class GpuEmbeddingStage {

    /**
     * A vocabulary-parallel token lookup (vLLM's {@code VocabParallelEmbedding}): every rank holds a row slice of the
     * token table, looks up the step's tokens that fall in its slice and zeroes the others, and a collective sums the
     * rows. The stage installs one on a tensor-parallel split; the rest of the embedding pipeline runs on the model
     * device over the summed rows.
     */
    public interface VocabParallel {
        /** First token-table row the model device holds. */
        int embeddingStart();

        /** Token-table rows the model device holds. */
        int embeddingCount();

        /** Rows of the whole token table. */
        int embeddingVocab();

        /**
         * Enqueues the summed token-table lookup of rows {@code [off, off + n)} of {@code tokens} into {@code out}.
         * The rows are {@code tokenWidth} floats each, and the result is summed across every rank before it returns.
         */
        void lookup(int[] tokens, int off, int n, ScratchSlot out);
    }

    /** Bit pattern of {@code 0.0f} for {@link GpuOps#fill}. */
    private static final int ZERO_BITS = 0;
    /** Temporary of the step's token ids for the LoRA deltas when the token-id slot does not start at element 0. */
    private static final String T_LORA_IDS = "embedding.lora_ids";
    /** Rows of the token lookup before {@code project_in}, when the table is narrower than {@code n_embd}. */
    private static final String T_TOKEN_ROWS = "embedding.token_rows";

    private final GpuForwardContext ctx;
    private final GpuOps ops;
    private final GpuWeight tokenTable;
    /** Vocabulary-parallel lookup, or null when the model device holds the whole token table. */
    private final VocabParallel vocabParallel;
    private final GpuWeight inputProjection;
    /** Columns of a token-table row; equals {@link #nEmbd} without a projection. */
    private final int tokenWidth;
    private final GpuWeight typeTable;
    private final GpuWeight posTable;
    private final GpuWeight normWeight;
    private final GpuWeight normBias;
    private final float normEps;
    /** Epsilon of the weightless input RMS norm; 0 when the model has none. */
    private final float inputRmsEps;
    /** Weight of the input RMS norm ({@code token_embd_norm} on Inkling), or null for the weightless norm. */
    private final GpuWeight inputRmsNormWeight;
    private final float scale;
    private final int nEmbd;
    private final int nEmbdInp;
    /** Added to every position before the learned position lookup (OPT's two padding rows). */
    private final int posOffset;
    private final int streamWidth;
    private final int hcStreams;
    private final boolean keepInput;
    private final ScratchSlot x;
    private final ScratchSlot work;
    private final ScratchSlot addend;
    private final ScratchSlot tokenIds;
    private final ScratchSlot inputEmbd;
    /** Token lookup rows of width {@link #tokenWidth}; null without a projection (the lookup writes {@link #work}). */
    private final ScratchSlot tokenRows;
    private final int ubatch;
    private final ByteBuffer idBytes;
    private final int[] hostIds;
    private final int[] hostTypes;
    private final int[] hostPositions;
    /** LoRA deltas of the token table, applied after the lookup and before the embedding scale. */
    private final GpuProjectionDeltas.EmbeddingSite embdDeltas;
    /** The step's token record: the token id of a token row, {@code -1} for an embedding row. */
    private final int[] recordIds;
    /** Host staging of a run of embedding rows ({@code ubatch * nEmbd} floats), allocated by the first one. */
    private float[] rowStaging;

    /**
     * Binds the stage to a model context.
     *
     * @throws UnsupportedOperationException when the model has no token table (drafters that embed through their
     *                                       target model)
     * @throws MissingTensorException        when a table the spec names is not resident
     * @throws IllegalArgumentException      when a table does not match {@code n_embd} or a scratch slot is too small
     */
    public GpuEmbeddingStage(GpuForwardContext ctx) {
        this(ctx, null);
    }

    /**
     * Binds the stage to a model context with a vocabulary-parallel token lookup: the token table holds only the
     * model device's row slice ({@link VocabParallel}), and the lookup sums every rank's rows before the remaining
     * pipeline runs.
     *
     * @throws IllegalArgumentException when the resident token table does not match the split's rank-0 slice
     */
    public GpuEmbeddingStage(GpuForwardContext ctx, VocabParallel vocabParallel) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.ops = ctx.ops();
        this.vocabParallel = vocabParallel;
        ModelSpec spec = ctx.spec();
        EmbeddingSpec e = spec.embedding();
        String arch = spec.arch().ggufName();
        if (e.tokenEmbd() == null) {
            throw new UnsupportedOperationException(arch + ": the model has no token table; it embeds through its"
                    + " target model");
        }
        GpuWeights weights = ctx.weights();
        this.nEmbd = e.nEmbd();
        this.nEmbdInp = Math.max(e.nEmbdInp(), e.nEmbd());
        this.tokenWidth = e.tokenWidth();
        this.posOffset = e.posOffset();
        this.tokenTable = table(weights, arch, e.tokenEmbd(), "token table", tokenWidth);
        if (vocabParallel != null && (vocabParallel.embeddingStart() != 0
                || tokenTable.rows() != vocabParallel.embeddingCount()
                || tokenTable.rows() > vocabParallel.embeddingVocab())) {
            throw new IllegalArgumentException(arch + ": the resident token table holds " + tokenTable.rows()
                    + " rows, not the vocabulary split's rank-0 slice [" + vocabParallel.embeddingStart() + ", "
                    + (vocabParallel.embeddingStart() + vocabParallel.embeddingCount()) + ") of "
                    + vocabParallel.embeddingVocab());
        }
        this.inputProjection = e.inputProjection() != null ? require(weights, arch, e.inputProjection().name()) : null;
        if (inputProjection != null && (inputProjection.cols() != tokenWidth || inputProjection.rows() != nEmbd)) {
            throw new IllegalArgumentException(arch + ": project_in is " + inputProjection.cols() + " x "
                    + inputProjection.rows() + ", expected " + tokenWidth + " x " + nEmbd);
        }
        this.typeTable = e.tokenTypes() != null
                ? table(weights, arch, e.tokenTypes(), "token-type table", nEmbd) : null;
        this.posTable = e.posEmbd() != null ? table(weights, arch, e.posEmbd(), "position table", nEmbd) : null;
        if (e.tokenEmbdNorm() != null) {
            this.normWeight = require(weights, arch, e.tokenEmbdNorm().name());
            this.normBias = weights.find(biasName(e.tokenEmbdNorm().name()));
        } else {
            this.normWeight = null;
            this.normBias = null;
        }
        this.normEps = spec.hparams().normEps();
        if (normWeight != null && !(normEps > 0f)) {
            throw new IllegalArgumentException(arch + ": token_embd_norm needs a positive norm epsilon, got "
                    + normEps);
        }
        this.inputRmsEps = e.inputRmsNorm() ? spec.hparams().normRmsEps() : 0f;
        if (e.inputRmsNorm() && !(inputRmsEps > 0f)) {
            throw new IllegalArgumentException(arch + ": the input RMS norm needs a positive RMS epsilon, got "
                    + inputRmsEps);
        }
        this.inputRmsNormWeight = e.inputRmsNormWeight() != null
                ? require(weights, arch, e.inputRmsNormWeight().name()) : null;
        this.scale = e.embdScale();
        LayerSpec first = spec.layers().get(0);
        this.streamWidth = first.residual().streamWidth();
        if (streamWidth % nEmbd != 0) {
            throw new IllegalArgumentException(arch + ": stream width " + streamWidth + " of layer 0 is not a multiple"
                    + " of n_embd " + nEmbd);
        }
        this.hcStreams = streamWidth / nEmbd;
        this.keepInput = nEmbdInp > nEmbd || readsEmbeddingAgain(spec);
        this.ubatch = ctx.scratch().ubatch();
        this.x = ctx.scratch().slot(ScratchSlots.X).require((long) ubatch * streamWidth);
        this.work = hcStreams > 1 ? ctx.scratch().slot(ScratchSlots.BRANCH).require((long) ubatch * nEmbd) : x;
        this.addend = ctx.scratch().slot(ScratchSlots.NORMED).require((long) ubatch * nEmbd);
        this.tokenIds = ctx.scratch().slot(ScratchSlots.TOKEN_IDS).require(2L * ubatch);
        this.inputEmbd = keepInput
                ? ctx.scratch().slot(ScratchSlots.INPUT_EMBD).require((long) ubatch * nEmbdInp) : null;
        this.tokenRows = inputProjection != null
                ? ctx.scratch().temp(T_TOKEN_ROWS, (long) ubatch * tokenWidth) : null;
        this.idBytes = ByteBuffer.allocate(2 * ubatch * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.hostIds = new int[ubatch];
        this.hostTypes = new int[ubatch];
        this.hostPositions = new int[ubatch];
        this.recordIds = new int[ubatch];
        this.embdDeltas = ctx.projectionDeltas().serveEmbedding(TensorNames.weight(TensorNames.TOKEN_EMBD),
                tokenTable.rows(), tokenWidth, null);
        if (tokenIds.offset() != 0) {
            ctx.scratch().temp(T_LORA_IDS, ubatch);
        }
    }

    /**
     * True when a layer reads the model input again: the {@code embd_skip} add, or a Zamba2 hybrid layer, whose
     * shared transformer block concatenates the input embedding with the residual stream.
     */
    private static boolean readsEmbeddingAgain(ModelSpec spec) {
        for (LayerSpec l : spec.layers()) {
            if (l.residual().embeddingSkip() || l.hasExtra(LayerSpec.Extras.EMBD_SKIP)
                    || l.zamba2() != null && l.zamba2().hybrid()) {
                return true;
            }
        }
        return false;
    }

    private static GpuWeight table(GpuWeights weights, String arch, TensorRef ref, String what, int width) {
        GpuWeight w = require(weights, arch, ref.name());
        if (w.cols() != width) {
            throw new IllegalArgumentException(arch + ": " + what + " " + w.name() + " has " + w.cols()
                    + " columns, expected " + width);
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

    /** {@code <base>.bias} for a {@code <base>.weight} name; null for any other name. */
    static String biasName(String weightName) {
        String suffix = ".weight";
        return weightName.endsWith(suffix)
                ? weightName.substring(0, weightName.length() - suffix.length()) + ".bias" : null;
    }

    /** Width of each {@link ScratchSlots#X} row the stage writes. */
    public int streamWidth() {
        return streamWidth;
    }

    /** Model width {@code n_embd}. */
    public int nEmbd() {
        return nEmbd;
    }

    /** Rows of the token table (the valid token-id range), the whole vocabulary on a split. */
    public long vocabRows() {
        return vocabParallel != null ? vocabParallel.embeddingVocab() : tokenTable.rows();
    }

    /**
     * Enqueues the embedding of {@code tokens[off .. off + n)} at positions {@code pos0 ..} into
     * {@link ScratchSlots#X}.
     *
     * @param types token types parallel to {@code tokens} (same indexing), or null for type 0
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, ubatch]}, a token id or type is outside its
     *                                  table, or a position exceeds the learned positions
     */
    public void embed(int[] tokens, int off, int n, int[] types, int pos0) {
        Objects.requireNonNull(tokens, "tokens");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("embedding step of " + n + " tokens outside [1, " + ubatch + "]");
        }
        if (off < 0 || off + n > tokens.length || (types != null && types.length < tokens.length)) {
            throw new IllegalArgumentException("token range [" + off + ", " + (off + n) + ") outside the "
                    + tokens.length + " tokens" + (types != null ? " / " + types.length + " types" : ""));
        }
        fillHost(tokens, off, n, types, pos0);
        embedRows(n, null);
    }

    /**
     * Enqueues the embedding of whole {@code tokens} rows at per-row {@code positions} (batched rows in position
     * order per sequence, not one consecutive run) into {@link ScratchSlots#X}.
     *
     * @param types token types parallel to {@code tokens}, or null for type 0
     * @throws IllegalArgumentException when the arrays disagree in length, a token count is outside
     *                                  {@code [1, ubatch]}, or an id, type or position is outside its table
     */
    public void embedBatched(int[] tokens, int[] positions, int[] types) {
        embedBatched(tokens, positions, types, null);
    }

    /**
     * {@link #embedBatched(int[], int[], int[])} over rows that are token rows or precomputed embedding rows: row
     * {@code t} is the embedding {@code rows[t]} when it is non-null ({@code n_embd} or the input width
     * {@code nEmbdInp} values; {@code tokens[t]} is ignored) and the lookup of {@code tokens[t]} otherwise. The stage
     * keeps no reference to the rows.
     *
     * @param rows one entry per row, null for a token row; null as a whole for a step of token rows
     * @throws IllegalArgumentException when the arrays disagree in length, the row count is outside
     *                                  {@code [1, ubatch]}, an embedding row has another width, or an id, type or
     *                                  position is outside its table
     */
    public void embedBatched(int[] tokens, int[] positions, int[] types, float[][] rows) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(positions, "positions");
        int n = tokens.length;
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("embedding step of " + n + " rows outside [1, " + ubatch + "]");
        }
        if (positions.length != n || (types != null && types.length != n) || (rows != null && rows.length != n)) {
            throw new IllegalArgumentException(n + " tokens with " + positions.length + " positions"
                    + (types != null ? ", " + types.length + " types" : "")
                    + (rows != null ? ", " + rows.length + " embedding entries" : ""));
        }
        fillHostBatched(tokens, positions, types, rows);
        embedRows(n, rows);
    }

    /**
     * Enqueues precomputed embedding rows {@code rows[off .. off + n)} at positions {@code pos0 ..} into
     * {@link ScratchSlots#X} (a contiguous {@code prompt_embeds} run); see the class notes for the row semantics.
     *
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, ubatch]}, the range is outside the rows,
     *                                  a row has another width, or a position exceeds the learned positions
     */
    public void embedRows(float[][] rows, int off, int n, int pos0) {
        Objects.requireNonNull(rows, "rows");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("embedding step of " + n + " rows outside [1, " + ubatch + "]");
        }
        if (off < 0 || (long) off + n > rows.length) {
            throw new IllegalArgumentException("row range [" + off + ", " + ((long) off + n) + ") outside the "
                    + rows.length + " rows");
        }
        float[][] step = new float[n][];
        int[] positions = new int[n];
        for (int t = 0; t < n; t++) {
            step[t] = Objects.requireNonNull(rows[off + t], "embedding row " + (off + t));
            positions[t] = pos0 + t;
        }
        embedBatched(new int[n], positions, null, step);
    }

    /** Whether a caller-provided embedding row of {@code width} values fits: {@code n_embd} or {@code nEmbdInp}. */
    public boolean acceptsRowWidth(int width) {
        return width == nEmbd || width == nEmbdInp;
    }

    /** Width of the {@link ScratchSlots#INPUT_EMBD} rows, the widest embedding row accepted ({@code nEmbdInp}). */
    public int inputWidth() {
        return nEmbdInp;
    }

    /**
     * The shared embedding pipeline once the host staging arrays hold the step; {@code rows} (null entries for token
     * rows, or null as a whole) replace the lookup of their rows.
     */
    private void embedRows(int n, float[][] rows) {
        ctx.scratch().recordStepTokens(recordIds, n);
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(-1, ExecutionObserver.Stage.EMBEDDING);
        }
        writeIds(n);
        int embeddingRows = countRows(rows, n);
        if (embeddingRows < n) {
            lookup(n, n * tokenWidth, inputProjection != null ? tokenRows : work);
            if (inputProjection != null) {
                ops.matmul(inputProjection, tokenRows.buffer(), tokenRows.offset(), work.buffer(), work.offset(), n);
            }
        }
        if (embeddingRows > 0) {
            writeEmbeddingRows(rows, n);
        }
        int workLen = n * nEmbd;
        if (typeTable != null) {
            ops.embed(typeTable, slice(hostTypes, n), addend.buffer(), addend.offset());
            ops.add(work.buffer(), work.offset(), addend.buffer(), addend.offset(), work.buffer(), work.offset(),
                    workLen);
        }
        if (posTable != null) {
            ops.embed(posTable, slice(hostPositions, n), addend.buffer(), addend.offset());
            ops.add(work.buffer(), work.offset(), addend.buffer(), addend.offset(), work.buffer(), work.offset(),
                    workLen);
        }
        if (normWeight != null) {
            ops.norm(NormKind.LAYER, normWeight, normBias, normEps, work.buffer(), work.offset(), work.buffer(),
                    work.offset(), n, nEmbd);
        }
        if (inputRmsEps > 0f) {
            ops.norm(NormKind.RMS, inputRmsNormWeight, null, inputRmsEps, work.buffer(), work.offset(),
                    work.buffer(), work.offset(), n, nEmbd);
        }
        if (keepInput) {
            keepInputRows(n);
            if (embeddingRows > 0 && nEmbdInp > nEmbd) {
                writeRowTails(rows, n);
            }
        }
        if (hcStreams > 1) {
            expandStreams(n);
        }
        if (observed) {
            obs.onStageEnd(-1, ExecutionObserver.Stage.EMBEDDING, x, n, streamWidth);
        }
    }

    /**
     * Steps 2-4 for every row: the table lookup, its LoRA deltas and the embedding scale, into {@code dst}
     * ({@link #tokenRows} width {@link #tokenWidth} before a projection, else {@link #work}).
     */
    private void lookup(int n, int rowsLen, ScratchSlot dst) {
        if (vocabParallel != null) {
            if (!embdDeltas.isEmpty()) {
                throw new UnsupportedOperationException("a vocabulary-parallel token table cannot serve LoRA"
                        + " deltas registered on " + tokenTable.name());
            }
            vocabParallel.lookup(hostIds, 0, n, dst);
        } else if (tokenTable.isSingleShard() && tokenIds.offset() == 0) {
            ops.embed(tokenTable, tokenIds.buffer(), n, dst.buffer(), dst.offset());
        } else {
            ops.embed(tokenTable, slice(hostIds, n), dst.buffer(), dst.offset());
        }
        if (vocabParallel == null && !embdDeltas.isEmpty()) {
            embdDeltas.apply(n, deviceIds(n), dst.buffer(), dst.offset(), tokenWidth);
        }
        if (scale != 1.0f) {
            ops.scale(dst.buffer(), dst.offset(), rowsLen, scale);
        }
    }

    private static int countRows(float[][] rows, int n) {
        int count = 0;
        if (rows != null) {
            for (int t = 0; t < n; t++) {
                count += rows[t] != null ? 1 : 0;
            }
        }
        return count;
    }

    /**
     * Writes the first {@code n_embd} values of every embedding row over its lookup row in the work rows, one
     * transfer per run of consecutive embedding rows.
     */
    private void writeEmbeddingRows(float[][] rows, int n) {
        if (rowStaging == null) {
            rowStaging = new float[Math.multiplyExact(ubatch, nEmbd)];
        }
        for (int t = 0; t < n; ) {
            if (rows[t] == null) {
                t++;
                continue;
            }
            int start = t;
            while (t < n && rows[t] != null) {
                System.arraycopy(rows[t], 0, rowStaging, (t - start) * nEmbd, nEmbd);
                t++;
            }
            long floats = (long) (t - start) * nEmbd;
            GpuEvent ev = ctx.queue().enqueueWriteBuffer(work.buffer(),
                    ((long) work.offset() + (long) start * nEmbd) * Float.BYTES, floats * Float.BYTES, rowStaging);
            if (ev != null) {
                ev.close();
            }
        }
    }

    /**
     * The tail {@code [n_embd, nEmbdInp)} of every full-width embedding row into its {@link ScratchSlots#INPUT_EMBD}
     * row, after the zero padding of {@link #keepInputRows}.
     */
    private void writeRowTails(float[][] rows, int n) {
        int tail = nEmbdInp - nEmbd;
        for (int t = 0; t < n; t++) {
            if (rows[t] == null || rows[t].length != nEmbdInp) {
                continue;
            }
            System.arraycopy(rows[t], nEmbd, rowStaging, 0, tail);
            GpuEvent ev = ctx.queue().enqueueWriteBuffer(inputEmbd.buffer(),
                    ((long) inputEmbd.offset() + (long) t * nEmbdInp + nEmbd) * Float.BYTES,
                    (long) tail * Float.BYTES, rowStaging);
            if (ev != null) {
                ev.close();
            }
        }
    }

    /** Row {@code t} of the embeddings into each of the {@code hc} streams of row {@code t} of {@code X}. */
    private void expandStreams(int n) {
        for (int h = 0; h < hcStreams; h++) {
            ops.copy2d(work.buffer(), work.offset(), nEmbd, x.buffer(), x.offset() + h * nEmbd, streamWidth, n, nEmbd);
        }
    }

    /** Validates the step's ids, types and positions into the host staging arrays. */
    private void fillHost(int[] tokens, int off, int n, int[] types, int pos0) {
        long vocab = vocabRows();
        long typeRows = typeTable != null ? typeTable.rows() : 0;
        long posRows = posTable != null ? posTable.rows() : 0;
        for (int t = 0; t < n; t++) {
            int id = tokens[off + t];
            if (id < 0 || id >= vocab) {
                throw new IllegalArgumentException("token id " + id + " at " + (off + t) + " is outside the " + vocab
                        + " rows of " + tokenTable.name());
            }
            hostIds[t] = id;
            recordIds[t] = id;
            int type = types != null ? types[off + t] : 0;
            if (typeTable != null && (type < 0 || type >= typeRows)) {
                throw new IllegalArgumentException("token type " + type + " at " + (off + t) + " is outside the "
                        + typeRows + " rows of " + typeTable.name());
            }
            hostTypes[t] = type;
            long pos = (long) pos0 + t + posOffset;
            if (posTable != null && (pos < 0 || pos >= posRows)) {
                throw new IllegalArgumentException("position " + pos + " is outside the " + posRows
                        + " learned position rows of " + posTable.name());
            }
            hostPositions[t] = (int) pos;
        }
    }

    /**
     * {@link #fillHost} over batched rows whose positions arrive per row; an embedding row ({@code rows[t]} non-null)
     * is checked for its width and takes type 0 and the lookup id 0, whose lookup row it replaces.
     */
    private void fillHostBatched(int[] tokens, int[] positions, int[] types, float[][] rows) {
        long vocab = vocabRows();
        long typeRows = typeTable != null ? typeTable.rows() : 0;
        long posRows = posTable != null ? posTable.rows() : 0;
        for (int t = 0; t < tokens.length; t++) {
            boolean embeddingRow = rows != null && rows[t] != null;
            int id = embeddingRow ? 0 : tokens[t];
            if (embeddingRow && !acceptsRowWidth(rows[t].length)) {
                throw new IllegalArgumentException("embedding row " + t + " holds " + rows[t].length
                        + " values, expected n_embd " + nEmbd + (nEmbdInp != nEmbd ? " or the model input width "
                        + nEmbdInp : ""));
            }
            if (id < 0 || id >= vocab) {
                throw new IllegalArgumentException("token id " + id + " at " + t + " is outside the " + vocab
                        + " rows of " + tokenTable.name());
            }
            hostIds[t] = id;
            recordIds[t] = embeddingRow ? -1 : id;
            int type = embeddingRow || types == null ? 0 : types[t];
            if (typeTable != null && (type < 0 || type >= typeRows)) {
                throw new IllegalArgumentException("token type " + type + " at " + t + " is outside the "
                        + typeRows + " rows of " + typeTable.name());
            }
            hostTypes[t] = type;
            int pos = positions[t] + posOffset;
            if (posTable != null && (pos < 0 || pos >= posRows)) {
                throw new IllegalArgumentException("position " + pos + " is outside the " + posRows
                        + " learned position rows of " + posTable.name());
            }
            hostPositions[t] = pos;
        }
    }

    /** One write of {@code [ids][types]} (types only when the model has a type table) to the token-id slot. */
    private void writeIds(int n) {
        int count = typeTable != null ? 2 * n : n;
        idBytes.clear();
        for (int t = 0; t < n; t++) {
            idBytes.putInt(hostIds[t]);
        }
        if (typeTable != null) {
            for (int t = 0; t < n; t++) {
                idBytes.putInt(hostTypes[t]);
            }
        }
        GpuEvent ev = ctx.queue().enqueueWriteBuffer(tokenIds.buffer(), (long) tokenIds.offset() * Float.BYTES,
                (long) count * Integer.BYTES, idBytes.array());
        if (ev != null) {
            ev.close();
        }
    }

    /**
     * The step's token ids on the device from element 0: the token-id slot when it starts there, else a copy written
     * from the host ids to a temporary.
     */
    private GpuBuffer deviceIds(int n) {
        if (tokenIds.offset() == 0) {
            return tokenIds.buffer();
        }
        ScratchSlot ids = ctx.scratch().temp(T_LORA_IDS, ubatch);
        idBytes.clear();
        for (int t = 0; t < n; t++) {
            idBytes.putInt(hostIds[t]);
        }
        GpuEvent ev = ctx.queue().enqueueWriteBuffer(ids.buffer(), (long) ids.offset() * Float.BYTES,
                (long) n * Integer.BYTES, idBytes.array());
        if (ev != null) {
            ev.close();
        }
        return ids.buffer();
    }

    /** Stage output into {@link ScratchSlots#INPUT_EMBD} as {@code [t][nEmbdInp]} rows, zero past {@code n_embd}. */
    private void keepInputRows(int n) {
        if (nEmbdInp > nEmbd) {
            ops.fill(inputEmbd.buffer(), (long) inputEmbd.offset() * Float.BYTES, (long) n * nEmbdInp * Float.BYTES,
                    ZERO_BITS);
            ops.copy2d(work.buffer(), work.offset(), nEmbd, inputEmbd.buffer(), inputEmbd.offset(), nEmbdInp, n, nEmbd);
        } else {
            ops.copy(work.buffer(), work.offset(), inputEmbd.buffer(), inputEmbd.offset(), n * nEmbd);
        }
    }

    private static int[] slice(int[] values, int n) {
        int[] out = new int[n];
        System.arraycopy(values, 0, out, 0, n);
        return out;
    }

    @Override
    public String toString() {
        return "GpuEmbeddingStage{" + tokenTable.name() + ", n_embd=" + nEmbd + (nEmbdInp > nEmbd ? ", inp=" + nEmbdInp
                : "") + (hcStreams > 1 ? ", hc=" + hcStreams : "") + (scale != 1.0f ? ", scale=" + scale : "") + '}';
    }
}
