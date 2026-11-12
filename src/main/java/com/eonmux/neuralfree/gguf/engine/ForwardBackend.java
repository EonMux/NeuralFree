package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.ModelMemory;

import java.util.Objects;

/**
 * Runs the model's forward passes for one sequence on one device (GPU or CPU). Implemented by
 * {@code GpuForwardBackend}, {@code CpuForwardBackend} and the encoder-decoder backends; driven by the generation
 * loop, the speculative decoder, the diffusion generator and the embedding / classifier heads.
 *
 * <p>Contract shared by every method:
 * <ul>
 *   <li>Token ids must lie in {@code [0, nVocab)}; positions are absolute sequence positions.</li>
 *   <li>Logits rows are {@code nVocab} floats. A returned array may be reused by the next call on this backend;
 *       callers may modify it in place and must copy what they keep across calls.</li>
 *   <li>Calls that write memory throw {@link ContextFullException} when the positions they need exceed the
 *       memory's context length, before any work is enqueued.</li>
 *   <li>A backend is used by one thread at a time.</li>
 * </ul>
 */
public interface ForwardBackend extends AutoCloseable {

    /**
     * Sequences the backend's memory can hold (llama.cpp {@code n_seq_max}). Default 1: contiguous single-sequence
     * memory around {@link #prefill} / {@link #decode}.
     */
    default int maxSequences() {
        return 1;
    }

    /**
     * Sequences one memoryless batch packs ({@link #hiddenBatch} or {@link #forwardBatch} on a backend planned
     * without memory): each a contiguous run of rows from position 0 that attends only its own rows
     * ({@link BatchSegments}), within {@link #memorylessBatchRows()} rows. Default 1: a memoryless batch holds one
     * sequence.
     */
    default int memorylessSequences() {
        return 1;
    }

    /**
     * Rows one packed memoryless batch holds (the ubatch a cache-less step runs at once) while
     * {@link #memorylessSequences()} is above 1; 0 (the default) when the backend packs no sequences.
     */
    default int memorylessBatchRows() {
        return 0;
    }

    /**
     * Why this backend refuses {@link #forwardBatch}, or null when it can run batches. The device backends keep the
     * default: their batches run over paged memory ({@code neuralfree.parallel > 1}), on {@code neuralfree.parallel
     * = 1} their memory holds one sequence. The heap backend runs a single-sequence batch over the plain contiguous
     * memory too, so it answers null.
     */
    default String batchRefusal() {
        return "multi-sequence batches need neuralfree.parallel > 1";
    }

    /**
     * Why this backend's {@link #forwardBatch} refuses embedding rows ({@code ForwardBatch.Builder.addEmbedding}:
     * {@code prompt_embeds}, media embeddings), or null when it writes them. The batch engine asks before it admits
     * a prompt with embedding rows, so the refusal reaches that request alone. Default: the token-row path of
     * {@link #forwardBatch} refuses them.
     */
    default String embeddingRowsRefusal() {
        return getClass().getSimpleName() + " runs token rows only";
    }

    /**
     * The per-sequence encoder of an encoder-decoder backend whose {@link #forwardBatch} runs several decoder
     * sequences, or null: decoder-only backends, and encoder-decoder backends that run one sequence. Default null.
     */
    default SequenceEncoder sequenceEncoder() {
        return null;
    }

    /**
     * Evaluates a {@link ForwardBatch} and returns the output rows of the batch in batch order: one logits row per
     * output row (its kinds, {@code ForwardBatch.kinds(int)}; a backend that produces no requested kind
     * refuses through {@link #batchRefusal()}).
     *
     * <p>Default: a single-sequence batch whose first position continues the held sequence
     * ({@code memory().sequence().nPast() == positions[0]}) and whose positions are consecutive runs
     * {@link #forwardRows} on; otherwise this backend refuses. Requests the default path cannot serve (several
     * sequences, embedding rows, extra RoPE sections, non-causal rows, adapter slots, kinds beyond
     * {@code LOGITS}) also refuse. A backend with multi-sequence memory replaces this default.</p>
     *
     * @throws UnsupportedOperationException carrying {@link #batchRefusal()} or the refusing reason
     */
    default float[][] forwardBatch(ForwardBatch batch) {
        Objects.requireNonNull(batch, "batch");
        if (!batch.isSingleSequence()) {
            throw new UnsupportedOperationException(batchRefusal());
        }
        int[] pos = batch.positions();
        if (memory() == null || memory().sequence().nPast() != pos[0]) {
            throw new UnsupportedOperationException(batchRefusal());
        }
        for (int i = 1; i < pos.length; i++) {
            if (pos[i] != pos[i - 1] + 1) {
                throw new UnsupportedOperationException("the default forward path runs consecutive positions; row "
                        + i + " holds " + pos[i] + " after " + pos[i - 1] + ": " + batchRefusal());
            }
        }
        if (batch.embeddings() != null) {
            throw new UnsupportedOperationException("the default " + ForwardBatch.class.getSimpleName()
                    + " path runs token rows only; embedding rows need a batched backend");
        }
        if (batch.ropePositions() != null && batch.ropePositions().length > 1) {
            throw new UnsupportedOperationException("the default forward path holds one RoPE position per token; "
                    + "a batch with " + batch.ropePositions().length + " sections needs a batched backend");
        }
        if (batch.nonCausal()) {
            throw new UnsupportedOperationException("the default forward path runs causal attention; a non-causal"
                    + " batch needs a batched backend");
        }
        for (int i = 0; i < batch.nTokens(); i++) {
            if (batch.produces(i, OutputKind.HIDDEN) || batch.produces(i, OutputKind.ROUTING)) {
                throw new UnsupportedOperationException("row " + i + " requests an output kind beyond "
                        + OutputKind.LOGITS + ": " + batchRefusal());
            }
        }
        String slotRefusal = batch.adapterSlotRefusal("the default forward path");
        if (slotRefusal != null) {
            throw new UnsupportedOperationException(slotRefusal + ": " + batchRefusal());
        }
        return forwardRows(batch.tokens(), pos[0], batch.outputRows());
    }

    /**
     * {@link #forwardBatch} with greedy selection per output row; default host argmax over the returned logits,
     * the device path of llama.cpp {@code n_outputs} rows.
     *
     * @return the id of the highest logit of each output row (lowest id on ties), in batch order
     */
    default int[] forwardBatchGreedy(ForwardBatch batch) {
        float[][] logits = forwardBatch(batch);
        int[] out = new int[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = argmax(logits[i]);
        }
        return out;
    }

    /**
     * Whether {@link #treeAcceptForward} walks a masked tree verify's acceptance on the device
     * ({@code LLM_TreeAccept} over the rows' greedy ids); the default false reads the verify rows back and walks
     * them on the host.
     */
    default boolean treeAcceptOnDevice() {
        return false;
    }

    /**
     * {@link #forwardBatch} of the one masked single-pass verify batch with its acceptance walked on the device:
     * the rows' greedy ids never leave the device, the {@code LLM_TreeAccept} kernel follows the accepted path
     * through the proposal's parent array over them, and only the accepted depth, the path and the two greedy ids
     * the round's emissions need come back ({@link TreeAccept}). The memory, the hidden tap and any open
     * {@link #openRecurrentTreeRound} see the call exactly as {@link #forwardBatch}.
     *
     * @param batch    the tree verify batch {@code TreeVerify.batch} builds: the pending token plus the proposal's
     *                 nodes (shorter trees padded) under the per-row ancestor masks
     * @param proposal the proposal the batch verifies, at least one node
     * @return the device's acceptance walk of the batch
     * @throws UnsupportedOperationException when {@link #treeAcceptOnDevice()} is false, or the batch is not the
     *                                       one masked step the walk runs over
     */
    default TreeAccept treeAcceptForward(ForwardBatch batch, DraftTree proposal) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " walks a tree verify's acceptance on"
                + " the host; the rows come back as logits (forwardBatch)");
    }

    /**
     * Whether {@link #forwardBatchTopK} computes the candidates on the device ({@code LLM_Row_TopK}); the default
     * reads the batch's rows back and selects on the host, so a batch engine keeps reading whole rows there.
     */
    default boolean batchTopKOnDevice() {
        return false;
    }

    /**
     * {@link #forwardBatch} with a top-k readback: per output row the {@code k} best candidates in rank order with
     * their logits, the rows a sampler chain whose selection fits the cone samples from exactly
     * ({@code Sampler#coneWidth}: a truncated chain, or a greedy chain whose bans and biases only lower a logit;
     * llama.cpp backend sampling {@code -bs}). GPU backends compute the candidates on the device when
     * {@link #batchTopKOnDevice()} holds; the default selects on the host over the batch's full rows. The memory
     * and the hidden tap see the call exactly as {@link #forwardBatch}.
     *
     * @throws IllegalArgumentException when {@code k} is below one
     */
    default RowTopK forwardBatchTopK(ForwardBatch batch, int k) {
        Objects.requireNonNull(batch, "batch");
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        return RowTopK.of(forwardBatch(batch), k);
    }

    /**
     * {@link #forwardBatchTopK(ForwardBatch, int)} whose rows also carry their log-sum-exp over the whole row
     * ({@link RowTopK#logSumExp}), the normalizer of a candidate's log-softmax: a caller that scores candidates by
     * their probability under the row (a tree drafter's path scores) reads {@code k} values and one normalizer per
     * row instead of the vocabulary. GPU backends take the denominator their readout computes; the default reads
     * the full rows and computes both on the host.
     *
     * @throws IllegalArgumentException when {@code k} is below one, or a row holds no finite logit
     */
    default RowTopK forwardBatchTopKNormalized(ForwardBatch batch, int k) {
        Objects.requireNonNull(batch, "batch");
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        return RowTopK.ofNormalized(forwardBatch(batch), k);
    }

    /**
     * Whether {@link #forwardBatchTopK(ForwardBatch, int, LogitPrefilter)} applies the prefilter on the device
     * ({@code LLM_LogitsPrefilter}); the default false refuses a non-empty prefilter, so the caller keeps the
     * whole-row path.
     */
    default boolean batchPrefilterOnDevice() {
        return false;
    }

    /**
     * {@link #forwardBatchTopK(ForwardBatch, int)} with per-row pre-top-k logit edits applied on the device before
     * the readout: the fixed logit-bias table, the penalties and DRY ({@link LogitPrefilter}). {@code prefilter}
     * holds one row per output row of the batch, so the readback is the top-k of the edited rows exactly. The
     * default refuses a non-empty prefilter ({@link #batchPrefilterOnDevice()}).
     *
     * @throws UnsupportedOperationException when this backend cannot apply the prefilter on the device
     */
    default RowTopK forwardBatchTopK(ForwardBatch batch, int k, LogitPrefilter prefilter) {
        Objects.requireNonNull(batch, "batch");
        if (prefilter != null && !prefilter.isEmpty()) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " applies no logit prefilter on"
                    + " the device: " + batchRefusal());
        }
        return forwardBatchTopK(batch, k);
    }

    /**
     * {@link #forwardBatchTopK(ForwardBatch, int, LogitPrefilter)} that also reads the whole logits of the output rows
     * {@code fullRows} flags: one forward serves the rows a sampler chain samples from a cone and the rows that need
     * the whole vocabulary (a grammar, a logprobs capture, a verify round). {@code prefilter} (null for none) holds
     * one row per output row and must be empty on every flagged row, which therefore comes back unedited; the
     * candidates of a flagged row are computed too and carry no meaning for the caller. The default reads every row
     * back ({@link #forwardBatch}) and selects on the host, refusing a non-empty prefilter as
     * {@link #forwardBatchTopK(ForwardBatch, int, LogitPrefilter)} does.
     *
     * @throws IllegalArgumentException      when {@code k} is below one, {@code fullRows} does not hold one flag per
     *                                       output row, or a flagged row's prefilter row is not empty
     * @throws UnsupportedOperationException when this backend cannot apply the prefilter on the device
     */
    default SplitReadout forwardBatchSplit(ForwardBatch batch, boolean[] fullRows, int k, LogitPrefilter prefilter) {
        checkSplit(batch, fullRows, k, prefilter);
        if (prefilter != null && !prefilter.isEmpty()) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " applies no logit prefilter on"
                    + " the device: " + batchRefusal());
        }
        float[][] rows = forwardBatch(batch);
        float[][] whole = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            if (fullRows[i]) {
                whole[i] = rows[i];
            }
        }
        return new SplitReadout(RowTopK.of(rows, k), whole);
    }

    /**
     * Checks a {@link #forwardBatchSplit} call: a positive {@code k}, one flag per output row of {@code batch}, and
     * an empty prefilter row on every flagged row.
     */
    static void checkSplit(ForwardBatch batch, boolean[] fullRows, int k, LogitPrefilter prefilter) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(fullRows, "fullRows");
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        int outputs = batch.outputRows().length;
        if (fullRows.length != outputs) {
            throw new IllegalArgumentException(fullRows.length + " whole-row flags for " + outputs + " output rows");
        }
        if (prefilter != null) {
            if (prefilter.rows() != outputs) {
                throw new IllegalArgumentException("the prefilter holds " + prefilter.rows() + " rows for "
                        + outputs + " output rows");
            }
            for (int i = 0; i < outputs; i++) {
                if (fullRows[i] && !prefilter.row(i).isEmpty()) {
                    throw new IllegalArgumentException("output row " + i + " is read whole but carries prefilter"
                            + " edits");
                }
            }
        }
    }

    /**
     * Evaluates a {@link ForwardBatch} like {@link #forwardBatch} and returns the final hidden states of its output
     * rows in batch order: the rows after the output norm, the input of the LM, pooling and classifier heads
     * (llama.cpp {@code t_embd}). Default: refused.
     *
     * @throws UnsupportedOperationException carrying {@link #batchRefusal()}
     */
    default float[][] hiddenBatch(ForwardBatch batch) {
        throw new UnsupportedOperationException("hidden states of batch rows are not produced by "
                + getClass().getSimpleName() + ": " + batchRefusal());
    }

    /**
     * The observer objects installed in the backend and its contexts (hidden taps, execution observers, matmul
     * overrides, diagnostics hooks): the observation inventory that prompt-row consumers must respect. Default:
     * {@link Observers#UNDECLARED}, which a consumer treats as observed (it skips no output row).
     */
    default Observers observers() {
        return Observers.UNDECLARED;
    }

    /**
     * Evaluates {@code tokens} at positions {@code pos0 .. pos0 + tokens.length - 1} (causal), chunked by the
     * memory's ubatch, storing them in memory.
     *
     * @return logits of the last row
     */
    float[] prefill(int[] tokens, int pos0);

    /**
     * Evaluates precomputed embedding rows at positions {@code pos0 .. pos0 + rows.length - 1} (causal) and
     * stores them in memory like {@link #prefill}: each row replaces the token-table lookup (vLLM
     * {@code prompt_embeds}, the chat {@code image_embeds}/{@code audio_embeds} parts), while the rest of the
     * input stage — the type and position additions and the input norm — runs as for tokens.
     *
     * @param rows one row per position, {@code n_embd} values (or the model's wider input width); the backend
     *             keeps no reference to them
     * @return logits of the last row
     * @throws UnsupportedOperationException when this backend has no embedding-row prefill (the CPU and GPU forward
     *                                       backends run it)
     */
    default float[] prefillEmbeddings(float[][] rows, int pos0) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not prefill embedding rows;"
                + " prompt_embeds needs an embedding-row backend (the CPU or GPU forward backend)");
    }

    /**
     * Evaluates one token at position {@code pos} and stores it in memory.
     *
     * @return logits of the token
     */
    float[] decode(int token, int pos);

    /**
     * {@link #decode} followed by greedy selection. GPU backends select on the device and read back only the
     * token id; the default selects on the host with {@link #argmax(float[])}.
     *
     * @return the id of the highest logit (lowest id on ties)
     */
    default int decodeGreedyOnDevice(int token, int pos) {
        return argmax(decode(token, pos));
    }

    /**
     * Whether {@link #chainGreedy} enqueues its step before the previous step's id reaches the host: the device
     * copies the previous step's greedy id into the next step's input, so the host detokenizes and checks stops for
     * one token while the device computes the next. True only while a chained step can be dropped again with
     * {@link #rollback}{@code (1)}; it may change between steps, and asking neither waits for nor completes an
     * outstanding step. The default false runs every greedy step to completion.
     */
    default boolean chainsGreedyOnDevice() {
        return false;
    }

    /**
     * Enqueues {@link #decodeGreedyOnDevice}{@code (token, pos)}: the token is stored at {@code pos} when the call
     * returns and the step's greedy id comes from {@link GreedyStep#id()}. The default runs the step at once.
     */
    default GreedyStep startGreedy(int token, int pos) {
        int id = decodeGreedyOnDevice(token, pos);
        return () -> id;
    }

    /**
     * Enqueues the greedy decode at {@code pos} of the id {@code previous} selects, the step greedy generation runs
     * after {@code previous}: that id is stored at {@code pos} when the call returns. When
     * {@link #chainsGreedyOnDevice()} holds, the step reads the id on the device and the call returns before
     * {@code previous} has finished; the default waits for {@code previous}'s id and decodes it at once. A caller
     * that does not keep the chained step (a stop the previous id triggers) drops its position again with
     * {@link #rollback}{@code (1)} when {@link #canRollback} allows; the row it leaves otherwise holds that id
     * computed exactly, as a decode of it would.
     *
     * @throws ContextFullException when {@code pos} is past the memory's context
     */
    default GreedyStep chainGreedy(GreedyStep previous, int pos) {
        Objects.requireNonNull(previous, "previous");
        return startGreedy(previous.id(), pos);
    }

    /**
     * Evaluates {@code tokens} causally at positions {@code pos0 ..} and stores them in memory, like
     * {@link #prefill}, returning the logits of the requested rows (speculative verification).
     *
     * @param outputRows row indices into {@code tokens} in the order the result lists them, or null for all rows
     * @return one logits row per requested row
     */
    float[][] forwardRows(int[] tokens, int pos0, int[] outputRows);

    /**
     * {@link #forwardRows} followed by greedy selection per requested row (speculative verification of an argmax
     * chain, llama.cpp backend sampling {@code -bs}). GPU backends select on the device and read back only the
     * ids; the default selects on the host with {@link #argmax(float[])} over the returned rows. The memory and the
     * hidden tap see the call exactly as {@link #forwardRows}.
     *
     * @param outputRows row indices into {@code tokens} in the order the result lists them, or null for all rows
     * @return the id of the highest logit of each requested row (lowest id on ties), in request order
     */
    default int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
        float[][] rows = forwardRows(tokens, pos0, outputRows);
        int[] ids = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            ids[i] = argmax(rows[i]);
        }
        return ids;
    }

    /**
     * Whether {@link #forwardRowsTopK} and {@link #decodeTopKOnDevice} compute their candidates on the device
     * ({@code LLM_Row_TopK}); the default reads the rows back and selects on the host.
     */
    default boolean topKOnDevice() {
        return false;
    }

    /**
     * {@link #forwardRows} with a top-k readback: per requested row the {@code k} best candidates in rank order
     * with their logits, the rows a sampler chain whose selection fits the cone samples from exactly
     * ({@code Sampler#coneWidth}: a truncated chain, or a greedy chain whose bans and biases only lower a logit;
     * llama.cpp backend sampling {@code -bs}). GPU backends compute the candidates on the device when
     * {@link #topKOnDevice()} holds; the default selects on the host over the full rows. The memory and the hidden
     * tap see the call exactly as {@link #forwardRows}.
     *
     * @param outputRows row indices into {@code tokens} in the order the result lists them, or null for all rows
     * @throws IllegalArgumentException when {@code k} is below one
     */
    default RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows) {
        return RowTopK.of(forwardRows(tokens, pos0, outputRows), k);
    }

    /**
     * Whether {@link #forwardRowsTopK(int[], int, int, int[], LogitPrefilter)} and
     * {@link #decodeTopKOnDevice(int, int, int, LogitPrefilter)} apply the prefilter on the device
     * ({@code LLM_LogitsPrefilter}); the default false refuses a non-empty prefilter.
     */
    default boolean topKPrefilterOnDevice() {
        return false;
    }

    /**
     * {@link #forwardRowsTopK(int[], int, int, int[])} with per-row pre-top-k logit edits applied on the device
     * before the readout ({@link LogitPrefilter}); {@code prefilter} holds one row per requested output row. The
     * default refuses a non-empty prefilter ({@link #topKPrefilterOnDevice()}).
     *
     * @throws UnsupportedOperationException when this backend cannot apply the prefilter on the device
     */
    default RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows, LogitPrefilter prefilter) {
        if (prefilter != null && !prefilter.isEmpty()) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " applies no logit prefilter on"
                    + " the device");
        }
        return forwardRowsTopK(tokens, pos0, k, outputRows);
    }

    /**
     * Evaluates {@code token} at {@code pos} and reads back its {@code k} best candidates in rank order, the
     * device-sampled decode of a chain whose selection fits the cone (llama.cpp backend sampling {@code -bs});
     * see {@link #forwardRowsTopK}. The memory and the hidden tap see the call exactly as {@link #decode}.
     */
    default RowTopK decodeTopKOnDevice(int token, int pos, int k) {
        return RowTopK.of(new float[][]{decode(token, pos)}, k);
    }

    /**
     * {@link #decodeTopKOnDevice(int, int, int)} with the row's pre-top-k logit edits applied on the device
     * before the readout ({@link LogitPrefilter}, one row). The default refuses a non-empty prefilter
     * ({@link #topKPrefilterOnDevice()}).
     *
     * @throws UnsupportedOperationException when this backend cannot apply the prefilter on the device
     */
    default RowTopK decodeTopKOnDevice(int token, int pos, int k, LogitPrefilter prefilter) {
        if (prefilter != null && !prefilter.isEmpty()) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " applies no logit prefilter on"
                    + " the device");
        }
        return decodeTopKOnDevice(token, pos, k);
    }

    /**
     * Evaluates the whole of {@code tokens} at positions {@code 0 .. tokens.length - 1} with non-causal attention,
     * without reading or modifying memory (masked diffusion, encoders).
     *
     * @param outputRows row indices into {@code tokens} in the order the result lists them, or null for all rows
     * @return one logits row per requested row
     */
    float[][] forwardAll(int[] tokens, int[] outputRows);

    /**
     * Evaluates the whole of {@code tokens} from position 0 with the model's own attention mode (non-causal for
     * encoders, causal for decoder embedding models), without reading or modifying memory, and returns the final
     * hidden states after the output norm: the input of the LM or classifier head (llama.cpp {@code t_embd}).
     *
     * @param tokenTypes segment ids per token (BERT {@code token_types}), or null for all zero
     * @return {@code [tokens.length][width]} hidden rows
     */
    float[][] hiddenRows(int[] tokens, int[] tokenTypes);

    /** Empties every cache and zeroes every state ({@link ModelMemory#reset()}). */
    void reset();

    /** Whether the last {@code nTokens} positions can be dropped ({@link ModelMemory#canRollback(int)}). */
    boolean canRollback(int nTokens);

    /**
     * Drops the last {@code nTokens} positions from memory.
     *
     * @throws IllegalStateException when {@link #canRollback(int)} is false
     */
    void rollback(int nTokens);

    /**
     * Context shift (llama.cpp {@code n_keep} / {@code n_discard}): keeps positions {@code [0, nKeep)}, drops
     * {@code [nKeep, nKeep + nDiscard)} and moves every later position down by {@code nDiscard}, re-rotating the
     * stored keys.
     *
     * @throws UnsupportedOperationException when a layer's memory cannot be shifted (recurrent and sparse mixers)
     */
    void shiftContext(int nKeep, int nDiscard);

    /**
     * Why the per-sequence context shift over a paged memory ({@link #shiftSequence}) would refuse, or null when
     * it can run. The engine asks once, at construction.
     */
    default String shiftSequenceRefusal() {
        return getClass().getSimpleName() + " shifts no paged sequence";
    }

    /**
     * The per-sequence context shift over a paged memory (llama.cpp server's slot shift,
     * server-context.cpp:3800-3808): removes the positions {@code [nKeep, nKeep + nDiscard)} of sequence
     * {@code seq} ({@code seq_rm}) and moves every later position down by {@code nDiscard} ({@code seq_add}),
     * re-rotating the moved keys.
     *
     * @throws UnsupportedOperationException per {@link #shiftSequenceRefusal}, or when the memory refuses the move
     *                                       for its storage form (a DeepSeek-V4 compressor beyond its rollback
     *                                       snapshots, recurrent state)
     */
    default void shiftSequence(int seq, int nKeep, int nDiscard) {
        throw new UnsupportedOperationException(shiftSequenceRefusal());
    }

    /**
     * Why the per-sequence position edits ({@link #addSequence}, {@link #divSequence}) would refuse, or null when
     * the memory can re-rotate their keys across sequences. The engine asks once, at construction.
     */
    default String sequenceEditRefusal() {
        return getClass().getSimpleName() + " edits no paged sequence";
    }

    /**
     * Moves the stored positions {@code [p0, p1)} of sequence {@code seq} by {@code delta} (llama.cpp
     * {@code llama_memory_seq_add}), re-rotating the moved keys: the self-extend's outer shifts and the server's
     * chunk reuse.
     *
     * @throws UnsupportedOperationException per {@link #sequenceEditRefusal()}, or when the memory refuses the move
     * @throws IllegalArgumentException      for a negative {@code p0} or a refused range
     */
    default void addSequence(int seq, int p0, int p1, int delta) {
        throw new UnsupportedOperationException(sequenceEditRefusal());
    }

    /**
     * Divides the stored positions {@code [p0, p1)} of sequence {@code seq} by {@code d} (llama.cpp
     * {@code llama_memory_seq_div}), re-rotating each moved key by its own shift: the position compression of
     * self-extend (group attention).
     *
     * @throws UnsupportedOperationException per {@link #sequenceEditRefusal()}, or when the memory refuses the edit
     * @throws IllegalArgumentException      for a negative {@code p0} or a divisor below 2
     */
    default void divSequence(int seq, int p0, int p1, int d) {
        throw new UnsupportedOperationException(sequenceEditRefusal());
    }

    /**
     * Re-appends the held rows of cached pages at the append cursor of sequence {@code seq} (llama.cpp server's KV
     * chunk reuse, the copy side of its {@code seq_rm} + {@code seq_add}): the run of {@code rows} rows starts at
     * source row {@code srcRow0} of the first page of {@code srcPages} (one page per KV group per run page, the
     * cache blocks' pages) and lands at the sequence's own slots, at positions past its stored ones; copied rows
     * re-rotate by the uniform {@code delta} between a source row's stored position and its new one, and a
     * {@code delta} 0 run whose rows meet the cursor page-aligned shares whole pages instead of copying them.
     *
     * @throws UnsupportedOperationException per {@link #sequenceEditRefusal()}, or when the memory refuses the edit
     * @throws com.eonmux.neuralfree.gguf.memory.MemoryFullException when the re-append needs pages the pools lack
     */
    default void reuseChunk(int seq, int[][] srcPages, int srcRow0, int pos0, int rows, int delta,
                            int[] tokens, int embeddingToken) {
        throw new UnsupportedOperationException(sequenceEditRefusal());
    }

    /**
     * Runs the page copies and row moves a sequence-level compaction queued on this backend's memory
     * ({@code SequenceMemory#moveRows}): their bytes land on the caches and the queue drains before the call
     * returns. A no-op when the memory keeps nothing queued, or carries its queued moves itself (the host memory
     * can defer them to its next step). The tree verify's commit runs it so the compaction is complete before the
     * bookkeeping's removal drops the queue.
     */
    default void carryRowMoves() {
    }

    /**
     * Why this backend's tree verify cannot run through a target whose layers carry recurrent state, or null when
     * the single-pass tree step covers every recurrent mixer of the model ({@link #openRecurrentTreeRound} then
     * serves the round). The default refuses: a backend that cannot snapshot-free the per-path state must not run
     * a masked tree forward against it.
     */
    default String recurrentTreeRefusal() {
        return getClass().getSimpleName() + " has no tree step through recurrent layers";
    }

    /**
     * Opens the recurrent side of one single-pass tree verify: {@code rows} rows of one masked forward, the first
     * {@code realRows} of them computed (the pending token plus the proposal's nodes, the rest padding). The
     * forward runs inside this round — the tree step keeps per-node state off the committed state — and the tree
     * commit closes it after {@code RecurrentTreeRound#commit} replayed the accepted rows. One round at a time.
     *
     * @throws UnsupportedOperationException when {@link #recurrentTreeRefusal()} refuses the model's recurrent
     *                                       layers (naming the mixer or kind the tree step does not cover)
     * @throws IllegalStateException         when a round is already open
     */
    default RecurrentTreeRound openRecurrentTreeRound(int rows, int realRows) {
        throw new UnsupportedOperationException(recurrentTreeRefusal());
    }

    /** The memory this backend writes; drafters that share the target's KV bind to it. */
    ModelMemory memory();

    /**
     * Installs the hidden-state observer used by every later forward call.
     *
     * @param tap the observer; {@link HiddenTap#NONE} removes it
     * @throws NullPointerException when {@code tap} is null
     */
    void setHiddenTap(HiddenTap tap);

    /** Releases every device and host resource this backend owns. */
    @Override
    void close();

    /**
     * Index of the largest value, the first one on ties; NaN never wins over an earlier value (llama.cpp
     * greedy sampler order).
     *
     * @throws IllegalArgumentException when {@code logits} is empty
     */
    static int argmax(float[] logits) {
        if (logits.length == 0) {
            throw new IllegalArgumentException("argmax of an empty logits row");
        }
        int best = 0;
        float bestValue = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > bestValue) {
                bestValue = logits[i];
                best = i;
            }
        }
        return best;
    }
}
