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
 *   <li>{@code x += token_types[type]} (models/bert.cpp:90-94; a null type array selects row 0 as llama.cpp
 *       does) and {@code x += position_embd[pos]} (bert.cpp:95-97, gpt2.cpp);</li>
 *   <li>{@code token_embd_norm}: LayerNorm with its bias (bert.cpp:101, bloom.cpp:76-81, rwkv6.cpp:102), or the
 *       weightless RMS input norm with {@code f_norm_rms_eps} ({@link EmbeddingSpec#inputRmsNorm()}, talkie.cpp:50-51,
 *       muse-spark.cpp:86-87);</li>
 *   <li>hyper-connection models repeat each row over the {@code hc} streams of the first layer
 *       (deepseek4.cpp:1240-1242).</li>
 * </ol>
 * {@link ScratchSlots#INPUT_EMBD} receives the stage output as {@code [t][nEmbdInp]} rows when the model reads it
 * again: zero-padded past {@code n_embd} for deepstack models (llama-graph.cpp {@code ggml_pad}, then a view of the
 * first {@code n_embd} columns as the stream, so the deepstack slices of token rows are zero) and as the source of
 * the {@code embd_skip} layer extra.
 *
 * <p>Per step the stage allocates no device memory and never waits on the device. Not thread-safe.</p>
 */
public final class GpuEmbeddingStage {

    /** Bit pattern of {@code 0.0f} for {@link GpuOps#fill}. */
    private static final int ZERO_BITS = 0;
    /** Temporary of the step's token ids for the LoRA deltas when the token-id slot does not start at element 0. */
    private static final String T_LORA_IDS = "embedding.lora_ids";

    private final GpuForwardContext ctx;
    private final GpuOps ops;
    private final GpuWeight tokenTable;
    private final GpuWeight typeTable;
    private final GpuWeight posTable;
    private final GpuWeight normWeight;
    private final GpuWeight normBias;
    private final float normEps;
    /** Epsilon of the weightless input RMS norm; 0 when the model has none. */
    private final float inputRmsEps;
    private final float scale;
    private final int nEmbd;
    private final int nEmbdInp;
    private final int streamWidth;
    private final int hcStreams;
    private final boolean keepInput;
    private final ScratchSlot x;
    private final ScratchSlot work;
    private final ScratchSlot addend;
    private final ScratchSlot tokenIds;
    private final ScratchSlot inputEmbd;
    private final int ubatch;
    private final ByteBuffer idBytes;
    private final int[] hostIds;
    private final int[] hostTypes;
    private final int[] hostPositions;
    /** LoRA deltas of the token table, applied after the lookup and before the embedding scale. */
    private final GpuProjectionDeltas.EmbeddingSite embdDeltas;

    /**
     * Binds the stage to a model context.
     *
     * @throws UnsupportedOperationException when the model has no token table (drafters that embed through their
     *                                       target model)
     * @throws MissingTensorException        when a table the spec names is not resident
     * @throws IllegalArgumentException      when a table does not match {@code n_embd} or a scratch slot is too small
     */
    public GpuEmbeddingStage(GpuForwardContext ctx) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.ops = ctx.ops();
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
        this.tokenTable = table(weights, arch, e.tokenEmbd(), "token table");
        this.typeTable = e.tokenTypes() != null ? table(weights, arch, e.tokenTypes(), "token-type table") : null;
        this.posTable = e.posEmbd() != null ? table(weights, arch, e.posEmbd(), "position table") : null;
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
        this.idBytes = ByteBuffer.allocate(2 * ubatch * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.hostIds = new int[ubatch];
        this.hostTypes = new int[ubatch];
        this.hostPositions = new int[ubatch];
        this.embdDeltas = ctx.projectionDeltas().serveEmbedding(TensorNames.weight(TensorNames.TOKEN_EMBD),
                tokenTable.rows(), nEmbd, null);
        if (tokenIds.offset() != 0) {
            ctx.scratch().temp(T_LORA_IDS, ubatch);
        }
    }

    /** True when a layer adds the embedding to its output ({@code embd_skip}). */
    private static boolean readsEmbeddingAgain(ModelSpec spec) {
        for (LayerSpec l : spec.layers()) {
            if (l.residual().embeddingSkip() || l.hasExtra(LayerSpec.Extras.EMBD_SKIP)) {
                return true;
            }
        }
        return false;
    }

    private GpuWeight table(GpuWeights weights, String arch, TensorRef ref, String what) {
        GpuWeight w = require(weights, arch, ref.name());
        if (w.cols() != nEmbd) {
            throw new IllegalArgumentException(arch + ": " + what + " " + w.name() + " has " + w.cols()
                    + " columns, n_embd is " + nEmbd);
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

    /** Rows of the token table (the valid token-id range). */
    public long vocabRows() {
        return tokenTable.rows();
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
        ctx.scratch().recordStepTokens(hostIds, n);
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStageStart(-1, ExecutionObserver.Stage.EMBEDDING);
        }
        writeIds(n);
        int rowsLen = n * nEmbd;
        if (tokenTable.isSingleShard() && tokenIds.offset() == 0) {
            ops.embed(tokenTable, tokenIds.buffer(), n, work.buffer(), work.offset());
        } else {
            ops.embed(tokenTable, slice(hostIds, n), work.buffer(), work.offset());
        }
        if (!embdDeltas.isEmpty()) {
            embdDeltas.apply(n, deviceIds(n), work.buffer(), work.offset(), nEmbd);
        }
        if (scale != 1.0f) {
            ops.scale(work.buffer(), work.offset(), rowsLen, scale);
        }
        if (typeTable != null) {
            ops.embed(typeTable, slice(hostTypes, n), addend.buffer(), addend.offset());
            ops.add(work.buffer(), work.offset(), addend.buffer(), addend.offset(), work.buffer(), work.offset(),
                    rowsLen);
        }
        if (posTable != null) {
            ops.embed(posTable, slice(hostPositions, n), addend.buffer(), addend.offset());
            ops.add(work.buffer(), work.offset(), addend.buffer(), addend.offset(), work.buffer(), work.offset(),
                    rowsLen);
        }
        if (normWeight != null) {
            ops.norm(NormKind.LAYER, normWeight, normBias, normEps, work.buffer(), work.offset(), work.buffer(),
                    work.offset(), n, nEmbd);
        }
        if (inputRmsEps > 0f) {
            ops.norm(NormKind.RMS, null, null, inputRmsEps, work.buffer(), work.offset(), work.buffer(),
                    work.offset(), n, nEmbd);
        }
        if (keepInput) {
            keepInputRows(n);
        }
        if (hcStreams > 1) {
            expandStreams(n);
        }
        if (observed) {
            obs.onStageEnd(-1, ExecutionObserver.Stage.EMBEDDING, x, n, streamWidth);
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
        long vocab = tokenTable.rows();
        long typeRows = typeTable != null ? typeTable.rows() : 0;
        long posRows = posTable != null ? posTable.rows() : 0;
        for (int t = 0; t < n; t++) {
            int id = tokens[off + t];
            if (id < 0 || id >= vocab) {
                throw new IllegalArgumentException("token id " + id + " at " + (off + t) + " is outside the " + vocab
                        + " rows of " + tokenTable.name());
            }
            hostIds[t] = id;
            int type = types != null ? types[off + t] : 0;
            if (typeTable != null && (type < 0 || type >= typeRows)) {
                throw new IllegalArgumentException("token type " + type + " at " + (off + t) + " is outside the "
                        + typeRows + " rows of " + typeTable.name());
            }
            hostTypes[t] = type;
            long pos = (long) pos0 + t;
            if (posTable != null && (pos < 0 || pos >= posRows)) {
                throw new IllegalArgumentException("position " + pos + " is outside the " + posRows
                        + " learned positions of " + posTable.name());
            }
            hostPositions[t] = (int) pos;
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
