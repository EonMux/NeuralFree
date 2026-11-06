package com.eonmux.neuralfree.gguf.cpu.head;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.RopeSectionSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * The model-input stage of the CPU forward pass, the twin of {@code GpuEmbeddingStage}. Per token, following
 * llama.cpp {@code build_inp_embd} (llama-graph.cpp) and the BERT / GPT-2 graphs (models/bert.cpp:78-101):
 * <ol>
 *   <li>{@code token_embd[id]} decoded from any tensor type, with the table's inverse PRISM Hadamard
 *       ({@link CpuWeights#embedRow});</li>
 *   <li>{@code + scale * B A[id]}: the LoRA deltas registered on {@code token_embd.weight}
 *       ({@link CpuProjectionDeltas#serveEmbedding}, {@code build_inp_embd} llama-graph.cpp:2694-2709);</li>
 *   <li>{@code * embdScale} on the looked-up row, before any addition;</li>
 *   <li>{@code project_in · x} when {@link EmbeddingSpec#inputProjection()} maps a token table narrower than
 *       {@code n_embd} (OPT's {@code word_embed_proj_dim});</li>
 *   <li>{@code + token_types[type]} (type 0 when no types are given, as llama.cpp's "sentence A") and
 *       {@code + position_embd[pos + offset]} when the tables exist ({@link EmbeddingSpec#posOffset()});</li>
 *   <li>{@code token_embd_norm}: LayerNorm with its bias when present, epsilon {@code f_norm_eps} (every llama.cpp
 *       graph with this tensor builds it as {@code LLM_NORM}: bert, modern-bert, rwkv6, rwkv7), or the weightless
 *       RMS input norm with {@code f_norm_rms_eps} ({@link EmbeddingSpec#inputRmsNorm()}, talkie.cpp:50-51,
 *       muse-spark.cpp:86-87);</li>
 *   <li>zero padding from {@code n_embd} to {@link EmbeddingSpec#nEmbdInp()}.</li>
 * </ol>
 *
 * <p>Outputs of one step of {@code n} tokens:</p>
 * <ul>
 *   <li>{@link CpuScratch#INPUT_EMBD}: rows {@code [t][nEmbdInp]}, the model input before it enters the residual
 *       stream (read by the embedding-skip extra; per-layer-embedding extras may reuse the slot after layer 0's
 *       {@code beforeBlock});</li>
 *   <li>the residual stream {@code x}: rows {@code [t][streamWidth]} of layer 0. The trunk reads {@code n_embd}
 *       values per token (llama.cpp views the padded input back to {@code n_embd}); with hyper-connections the
 *       row is repeated over the {@code hc} streams, {@code x[(t*hc + h)*n_embd + i]} (deepseek4.cpp:1240-1243,
 *       glm5-next.cpp:709-711, qwen4exp.cpp:442-444);</li>
 *   <li>{@link CpuScratch#TOKEN_IDS}: ids at {@code [0, n)} and token types at {@code [n, 2n)};</li>
 *   <li>{@link CpuScratch#POS}: the four-section positions, section {@code s} of token {@code t} at
 *       {@code [s*n + t]} ({@link RopePositions#fillStep}); single-section layers read section 0.</li>
 * </ul>
 *
 * <p>Embedding-row steps replace the token-table lookup with caller-provided rows: {@link #embedRows} for a
 * contiguous run at consecutive positions (vLLM {@code prompt_embeds}), {@link #embedMediaRows} for a batched step
 * at explicit positions and sections (a media chunk, llama.cpp {@code mtmd}). A row holds {@code n_embd} values
 * (zero-padded to the input width) or the full input width {@code nEmbdInp} (a deepstack model's projected media
 * row); the token ids take the null token {@code -1} (llama.cpp {@code LLAMA_TOKEN_NULL}), and the type and
 * position additions and the input norm run on the row's first {@code n_embd} values as for tokens (llama.cpp
 * builds them on the output of {@code build_inp_embd}, whichever input it selected).</p>
 *
 * <p>Not thread-safe; one forward pass at a time. A step allocates nothing.</p>
 */
public final class CpuEmbeddingStage {

    private static final String TEMP_ROW = "CpuEmbeddingStage.row";
    /** Tokens per parallel task of the lookup. */
    private static final int TOKEN_GRAIN = 4;

    private final ModelHparams hp;
    private final CpuWeights weights;
    private final CpuMatVec matVec;
    private final CpuScratch scratch;
    private final EmbeddingSpec embd;
    private final CpuTensorRef tokenTable;
    private final CpuTensorRef inputProjection;
    private final CpuTensorRef typeTable;
    private final CpuTensorRef posTable;
    private final float[] normWeight;
    private final float[] normBias;
    /** Weight of the input RMS norm ({@code token_embd_norm} on Inkling), or null for the weightless norm. */
    private final float[] inputRmsNormWeight;
    private final int nEmbd;
    private final int nEmbdInp;
    /** Columns of a token-table row: the projection's input width, else {@code n_embd}. */
    private final int tokenWidth;
    /** Floats per {@link #rowTemp} row: the wider of {@code n_embd} and {@link #tokenWidth}. */
    private final int rowWidth;
    private final int posOffset;
    private final int streamWidth;
    private final int streams;
    private final float[] rowTemp;
    /** LoRA deltas of the token table, applied after the lookup and before the embedding scale. */
    private final CpuProjectionDeltas.EmbeddingSite embdDeltas;

    /**
     * @throws IllegalArgumentException when the embedding tables disagree with {@code n_embd} or the first layer's
     *                                  stream width is not a multiple of it
     */
    public CpuEmbeddingStage(CpuForwardContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        ModelSpec spec = ctx.spec();
        this.hp = spec.hparams();
        this.weights = ctx.weights();
        this.matVec = ctx.matVec();
        this.scratch = ctx.scratch();
        this.embd = Objects.requireNonNull(spec.embedding(), "embedding spec");
        this.nEmbd = embd.nEmbd();
        this.nEmbdInp = Math.max(embd.nEmbdInp(), nEmbd);
        this.streamWidth = spec.layer(0).residual().streamWidth();
        if (nEmbd < 1 || streamWidth % nEmbd != 0) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": first-layer stream width " + streamWidth
                    + " is not a multiple of n_embd " + nEmbd);
        }
        this.streams = streamWidth / nEmbd;
        this.tokenWidth = embd.tokenWidth();
        this.rowWidth = Math.max(nEmbd, tokenWidth);
        this.posOffset = embd.posOffset();
        this.tokenTable = table(embd.tokenEmbd(), "token_embd", tokenWidth);
        this.inputProjection = embd.inputProjection() != null ? weights.get(embd.inputProjection()) : null;
        if (inputProjection != null && (inputProjection.cols() != tokenWidth || inputProjection.rows() != nEmbd)) {
            throw new IllegalArgumentException("project_in is " + inputProjection.cols() + " x "
                    + inputProjection.rows() + ", expected " + tokenWidth + " x " + nEmbd);
        }
        this.typeTable = table(embd.tokenTypes(), "token_types", nEmbd);
        this.posTable = table(embd.posEmbd(), "position_embd", nEmbd);
        TensorRef norm = embd.tokenEmbdNorm();
        this.normWeight = norm != null ? weights.floats(norm) : null;
        this.normBias = norm != null ? biasOf(norm) : null;
        if (normWeight != null && normWeight.length != nEmbd) {
            throw new IllegalArgumentException("token_embd_norm has " + normWeight.length + " values, expected n_embd "
                    + nEmbd);
        }
        if (normBias != null && normBias.length != nEmbd) {
            throw new IllegalArgumentException("token_embd_norm bias has " + normBias.length
                    + " values, expected n_embd " + nEmbd);
        }
        this.inputRmsNormWeight = embd.inputRmsNormWeight() != null ? weights.floats(embd.inputRmsNormWeight())
                : null;
        if (inputRmsNormWeight != null && inputRmsNormWeight.length != nEmbd) {
            throw new IllegalArgumentException(embd.inputRmsNormWeight().name() + " has " + inputRmsNormWeight.length
                    + " values, expected n_embd " + nEmbd);
        }
        this.rowTemp = scratch.temp(TEMP_ROW, Math.multiplyExact(scratch.ubatch(), rowWidth));
        this.embdDeltas = tokenTable == null ? null : ctx.projectionDeltas().serveEmbedding(
                TensorNames.weight(TensorNames.TOKEN_EMBD), tokenTable.rows(), tokenWidth, null);
    }

    private CpuTensorRef table(TensorRef ref, String what, int width) {
        if (ref == null) {
            return null;
        }
        CpuTensorRef t = weights.get(ref);
        if (t.cols() != width) {
            throw new IllegalArgumentException(what + " rows hold " + t.cols() + " values, expected " + width);
        }
        return t;
    }

    private float[] biasOf(TensorRef weight) {
        String name = weight.name();
        String suffix = ".weight";
        if (!name.endsWith(suffix)) {
            return null;
        }
        return weights.floatsOrNull(name.substring(0, name.length() - suffix.length()) + ".bias");
    }

    /** Width of the residual rows this stage writes ({@code n_embd}, or {@code hc * n_embd}). */
    public int streamWidth() {
        return streamWidth;
    }

    /** Width of the {@link CpuScratch#INPUT_EMBD} rows ({@code max(n_embd, nEmbdInp)}). */
    public int inputWidth() {
        return nEmbdInp;
    }

    /**
     * Checks every input of a call before its first step: each token id against the token table, each type
     * against the type table, and positions {@code pos0 .. pos0 + tokens.length - 1} against the position table,
     * so a rejected call has not written memory.
     *
     * @param tokenTypes segment ids aligned with {@code tokens}, or null for type 0
     * @throws IllegalArgumentException when a value is outside its table or the arrays disagree in length
     * @throws IllegalStateException    when the model has no token table (drafters embed through their target)
     */
    public void checkInputs(int[] tokens, int[] tokenTypes, int pos0) {
        Objects.requireNonNull(tokens, "tokens");
        if (tokenTable == null) {
            throw new IllegalStateException(hp.arch().ggufName()
                    + " has no token_embd table; its inputs come from the target model");
        }
        if (tokenTypes != null && tokenTypes.length != tokens.length) {
            throw new IllegalArgumentException(tokenTypes.length + " token types for " + tokens.length + " tokens");
        }
        long end = (long) pos0 + tokens.length;
        if (pos0 < 0 || end > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("positions [" + pos0 + ", " + end + ") out of range");
        }
        for (int id : tokens) {
            checkRow("token id", id, tokenTable);
        }
        if (typeTable != null && tokenTypes != null) {
            for (int type : tokenTypes) {
                checkRow("token type", type, typeTable);
            }
        }
        if (posTable != null && tokens.length > 0) {
            checkRow("position", (int) (end - 1) + posOffset, posTable);
        }
    }

    /**
     * Embeds tokens {@code tokens[off .. off + n)} at positions {@code pos0 ..} into the residual stream {@code x}
     * and the step's scratch inputs.
     *
     * @param tokenTypes segment ids aligned with {@code tokens}, or null for type 0
     * @throws IllegalArgumentException  when {@code n} is outside {@code [1, ubatch]}, a range is outside its array,
     *                                   or a token id, type or position is outside its table
     * @throws IllegalStateException     when the model has no token table (drafters embed through their target)
     */
    public void embed(int[] tokens, int[] tokenTypes, int off, int n, int pos0, float[] x) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(x, "x");
        if (tokenTable == null) {
            throw new IllegalStateException(hp.arch().ggufName()
                    + " has no token_embd table; its inputs come from the target model");
        }
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("step of " + n + " tokens outside [1, " + scratch.ubatch() + "]");
        }
        if (off < 0 || (long) off + n > tokens.length) {
            throw new IllegalArgumentException("token range [" + off + ", " + ((long) off + n) + ") outside "
                    + tokens.length + " tokens");
        }
        if (tokenTypes != null && tokenTypes.length != tokens.length) {
            throw new IllegalArgumentException(tokenTypes.length + " token types for " + tokens.length + " tokens");
        }
        if (pos0 < 0 || (long) pos0 + n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("positions [" + pos0 + ", " + ((long) pos0 + n) + ") out of range");
        }
        if ((long) n * streamWidth > x.length) {
            throw new IllegalArgumentException("residual stream holds " + x.length + " floats, need "
                    + (long) n * streamWidth);
        }
        writeStepInputs(tokens, tokenTypes, off, n, pos0);
        float[] inp = scratch.slot(CpuScratch.INPUT_EMBD);
        int[] stepIds = scratch.ints(CpuScratch.TOKEN_IDS);
        CpuParallel.parallelFor(n, TOKEN_GRAIN, (from, to) -> {
            for (int t = from; t < to; t++) {
                embedToken(t, pos0 + t, stepIds, tokenTypes != null ? tokenTypes[off + t] : 0, inp);
            }
        });
        copyToStream(n, x);
    }

    /**
     * Writes media embedding rows into the residual stream and the step's scratch inputs: row {@code off + t} of
     * {@code rows} (a projected media embedding of {@code n_embd} or {@code nEmbdInp} values, copied in) becomes
     * row {@code t} and runs the input-row tail ({@link #embedRows} explains the row semantics), and the position
     * sections come from {@code ropeSections} when it is given (exactly the model's section count per row, the
     * {@code ForwardBatch.ropePositions} layout) or from the text pattern over {@code rowPos} otherwise.
     *
     * @param rowPos       sequence position of every row, also the mask position of a single-section model
     * @param ropeSections section-major positions {@code [section][n]} covering exactly {@code n} rows, or null
     * @throws IllegalArgumentException when a count or range is out of range, a row has another width than
     *                                  {@code n_embd} or {@code nEmbdInp} or the sections disagree with the
     *                                  model's rope type
     * @throws IllegalStateException    when the model has no token table (drafters embed through their target)
     */
    public void embedMediaRows(float[][] rows, int off, int n, int[] rowPos, int[][] ropeSections, float[] x) {
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(rowPos, "rowPos");
        Objects.requireNonNull(x, "x");
        if (tokenTable == null) {
            throw new IllegalStateException(hp.arch().ggufName()
                    + " has no token_embd table; its inputs come from the target model");
        }
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("step of " + n + " media rows outside [1, " + scratch.ubatch() + "]");
        }
        if (off < 0 || (long) off + n > rows.length) {
            throw new IllegalArgumentException("media row range [" + off + ", " + ((long) off + n) + ") outside "
                    + rows.length + " rows");
        }
        if (rowPos.length != n) {
            throw new IllegalArgumentException(rowPos.length + " positions for " + n + " media rows");
        }
        if ((long) n * streamWidth > x.length) {
            throw new IllegalArgumentException("residual stream holds " + x.length + " floats, need "
                    + (long) n * streamWidth);
        }
        checkRowWidths(rows, off, n);
        for (int t = 0; t < n; t++) {
            if (rowPos[t] < 0) {
                throw new IllegalArgumentException("media row " + t + " has negative position " + rowPos[t]);
            }
        }
        writeMediaPositions(rowPos, n, ropeSections);
        writeRows(rows, off, n, rowPos, 0, x);
    }

    /**
     * Writes a batched ubatch whose rows are token rows or embedding rows into the residual stream and the step's
     * scratch inputs: row {@code t} is the embedding {@code rows[t]} when it is non-null (the {@link #embedMediaRows}
     * row semantics, the null token in {@link CpuScratch#TOKEN_IDS}) and the token-table lookup of {@code tokens[t]}
     * otherwise (the {@link #embedBatched} row semantics). The position sections come from {@code ropeSections} when
     * it is given (exactly the model's section count per row), the text pattern over {@code rowPos} otherwise.
     *
     * @param rows   one entry per row: the embedding values, or null for a token row
     * @param rowPos sequence position of every row
     * @throws IllegalArgumentException when a count is out of range, an embedding row has another width than
     *                                  {@code n_embd} or {@code nEmbdInp}, a token id or position is out of range,
     *                                  or the sections disagree with the model's rope type
     * @throws IllegalStateException    when the model has no token table (drafters embed through their target)
     */
    public void embedMixedRows(int[] tokens, float[][] rows, int[] rowPos, int[][] ropeSections, float[] x) {
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(rowPos, "rowPos");
        checkMixedRows(tokens, rows, rowPos, x);
        int n = tokens.length;
        if (ropeSections == null) {
            RopePositions.fillStep(hp.mropeTextPositions(), rowPos, 0, n, scratch.ints(CpuScratch.POS));
        } else {
            fillSections(ropeSections, n, "rope");
        }
        float[] inp = scratch.slot(CpuScratch.INPUT_EMBD);
        int[] stepIds = scratch.ints(CpuScratch.TOKEN_IDS);
        CpuParallel.parallelFor(n, TOKEN_GRAIN, (from, to) -> {
            for (int t = from; t < to; t++) {
                float[] row = rows[t];
                if (row == null) {
                    embedToken(t, rowPos[t], stepIds, 0, inp);
                } else {
                    System.arraycopy(row, 0, inp, t * nEmbdInp, row.length);
                    finishInputRow(t, rowPos[t], 0, inp, row.length);
                }
            }
        });
        copyToStream(n, x);
    }

    /** Validates a mixed ubatch and writes its step ids: the token id of a token row, the null token otherwise. */
    private void checkMixedRows(int[] tokens, float[][] rows, int[] rowPos, float[] x) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(x, "x");
        if (tokenTable == null) {
            throw new IllegalStateException(hp.arch().ggufName()
                    + " has no token_embd table; its inputs come from the target model");
        }
        int n = tokens.length;
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("step of " + n + " rows outside [1, " + scratch.ubatch() + "]");
        }
        if (rows.length != n || rowPos.length != n) {
            throw new IllegalArgumentException(n + " rows with " + rows.length + " embedding entries and "
                    + rowPos.length + " positions");
        }
        if ((long) n * streamWidth > x.length) {
            throw new IllegalArgumentException("residual stream holds " + x.length + " floats, need "
                    + (long) n * streamWidth);
        }
        int[] ids = scratch.ints(CpuScratch.TOKEN_IDS);
        for (int t = 0; t < n; t++) {
            if (rows[t] != null) {
                checkRowWidths(rows, t, 1);
                ids[t] = -1;
            } else {
                checkRow("token id", tokens[t], tokenTable);
                ids[t] = tokens[t];
            }
            ids[n + t] = 0;
            if (rowPos[t] < 0) {
                throw new IllegalArgumentException("row " + t + " has negative position " + rowPos[t]);
            }
        }
    }

    /**
     * Fills the {@link CpuScratch#POS} slot of a media step: explicit sections when given (validated against the
     * model's positions per token), the four-section text pattern over {@code rowPos} otherwise, as every other
     * step of the stage fills it; single-section and rope-less models read section 0 only.
     */
    private void writeMediaPositions(int[] rowPos, int n, int[][] ropeSections) {
        int[] pos = scratch.ints(CpuScratch.POS);
        if (ropeSections == null) {
            RopePositions.fillStep(hp.mropeTextPositions(), rowPos, 0, n, pos);
            return;
        }
        fillSections(ropeSections, n, "media rope");
    }

    private void writeStepInputs(int[] tokens, int[] tokenTypes, int off, int n, int pos0) {
        int[] ids = scratch.ints(CpuScratch.TOKEN_IDS);
        for (int t = 0; t < n; t++) {
            ids[t] = tokens[off + t];
            ids[n + t] = tokenTypes != null ? tokenTypes[off + t] : 0;
        }
        RopePositions.fillStep(hp.mropeTextPositions(), pos0, n, scratch.ints(CpuScratch.POS));
    }

    /**
     * Embeds a batched ubatch's token rows at the explicit positions of {@code tokenPos} (batched steps over several
     * sequences carry non-consecutive positions, one segment per sequence) and fills the step inputs from them.
     *
     * @param tokenPos position of every ubatch row, aligned with {@code tokens}
     * @throws IllegalArgumentException  when a count or a position is out of range
     * @throws IllegalStateException     when the model has no token table (drafters embed through their target)
     */
    public void embedBatched(int[] tokens, int[] tokenPos, float[] x) {
        embedBatchedChecked(tokens, tokenPos, null, x);
        RopePositions.fillStep(hp.mropeTextPositions(), tokenPos, 0, tokens.length,
                scratch.ints(CpuScratch.POS));
        fillTokenBatchedRows(tokens, tokenPos, x);
    }

    /**
     * {@link #embedBatched} with explicit position sections: {@code ropeSections[s][t]} is section {@code s} of
     * token row {@code t} ({@code ForwardBatch.ropePositions} layout), validated against the model's positions per
     * token; null fills the sections from the text pattern over {@code tokenPos} as {@link #embedBatched} does.
     *
     * @throws IllegalArgumentException  when a count, range, position or section count is out of range
     * @throws IllegalStateException     when the model has no token table (drafters embed through their target)
     */
    public void embedBatched(int[] tokens, int[] tokenPos, int[][] ropeSections, float[] x) {
        embedBatchedChecked(tokens, tokenPos, null, x);
        if (ropeSections == null) {
            RopePositions.fillStep(hp.mropeTextPositions(), tokenPos, 0, tokens.length,
                    scratch.ints(CpuScratch.POS));
        } else {
            fillSections(ropeSections, tokens.length, "rope");
        }
        fillTokenBatchedRows(tokens, tokenPos, x);
    }

    /**
     * {@link #embedBatched(int[], int[], int[][], float[])} over rows that are token rows or precomputed embedding
     * rows: row {@code t} is the embedding {@code rows[t]} when it is non-null ({@code n_embd} or the input width
     * {@code nEmbdInp} values; {@code tokens[t]} is ignored and a memoryless batch carries -1 there) and the lookup
     * of {@code tokens[t]} otherwise (the vLLM V1 mixed {@code prompt_embeds} batch, llama.cpp's
     * {@code build_inp_embd} beside {@code build_inp_tokens}).
     *
     * @param rows one entry per row, null for a token row; null as a whole for a step of token rows
     * @throws IllegalArgumentException when the arrays disagree in length, the row count is outside
     *                                  {@code [1, ubatch]}, an embedding row has another width, or an id, position
     *                                  is outside its table
     * @throws IllegalStateException    when the model has no token table (drafters embed through their target)
     */
    public void embedBatched(int[] tokens, int[] tokenPos, float[][] rows, float[] x) {
        embedBatchedChecked(tokens, tokenPos, rows, x);
        RopePositions.fillStep(hp.mropeTextPositions(), tokenPos, 0, tokens.length,
                scratch.ints(CpuScratch.POS));
        fillMixedRows(tokens, tokenPos, rows, x);
    }

    /** Validates explicit section-major positions against the model's positions per token and fills the POS slot. */
    private void fillSections(int[][] ropeSections, int n, String what) {
        RopeSectionSupport.check(hp.ropeType(), ropeSections);
        int sections = com.eonmux.neuralfree.gguf.ops.rope.RopeMath.positionsPerToken(hp.ropeType());
        int[] pos = scratch.ints(CpuScratch.POS);
        for (int s = 0; s < sections; s++) {
            if (ropeSections[s] == null || ropeSections[s].length < n) {
                throw new IllegalArgumentException(what + " section " + s + " covers fewer than " + n + " rows");
            }
            System.arraycopy(ropeSections[s], 0, pos, s * n, n);
        }
    }


    /** The shared body of the batched token embeds: id checks, step inputs, lookup and stream replication. */
    private void embedBatchedChecked(int[] tokens, int[] tokenPos, float[][] rows, float[] x) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(tokenPos, "tokenPos");
        Objects.requireNonNull(x, "x");
        if (tokenTable == null) {
            throw new IllegalStateException(hp.arch().ggufName()
                    + " has no token_embd table; its inputs come from the target model");
        }
        int n = tokens.length;
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("step of " + n + " tokens outside [1, " + scratch.ubatch() + "]");
        }
        if (tokenPos.length != n) {
            throw new IllegalArgumentException(tokenPos.length + " positions for " + n + " tokens");
        }
        if ((long) n * streamWidth > x.length) {
            throw new IllegalArgumentException("residual stream holds " + x.length + " floats, need "
                    + (long) n * streamWidth);
        }
        int[] ids = scratch.ints(CpuScratch.TOKEN_IDS);
        for (int t = 0; t < n; t++) {
            if (rows != null && rows[t] != null) {
                if (!acceptsRowWidth(rows[t].length)) {
                    throw new IllegalArgumentException("embedding row " + t + " holds " + rows[t].length
                            + " values, expected n_embd " + nEmbd
                            + (nEmbdInp != nEmbd ? " or the model input width " + nEmbdInp : ""));
                }
                ids[t] = -1;
            } else {
                checkRow("token id", tokens[t], tokenTable);
                ids[t] = tokens[t];
            }
            ids[n + t] = 0;
        }
        for (int p : tokenPos) {
            if (p < 0) {
                throw new IllegalArgumentException("token position " + p + " is negative");
            }
        }
    }

    /** Token lookup and stream replication of a batched step, over the step inputs already written. */
    private void fillTokenBatchedRows(int[] tokens, int[] tokenPos, float[] x) {
        fillMixedRows(tokens, tokenPos, null, x);
    }

    /** Lookup or embedding row per step row, then the stream replication, over the step inputs already written. */
    private void fillMixedRows(int[] tokens, int[] tokenPos, float[][] rows, float[] x) {
        int n = tokens.length;
        float[] inp = scratch.slot(CpuScratch.INPUT_EMBD);
        int[] stepIds = scratch.ints(CpuScratch.TOKEN_IDS);
        CpuParallel.parallelFor(n, TOKEN_GRAIN, (from, to) -> {
            for (int t = from; t < to; t++) {
                float[] row = rows == null ? null : rows[t];
                if (row != null) {
                    System.arraycopy(row, 0, inp, t * nEmbdInp, row.length);
                    finishInputRow(t, tokenPos[t], 0, inp, row.length);
                } else {
                    embedToken(t, tokenPos[t], stepIds, 0, inp);
                }
            }
        });
        copyToStream(n, x);
    }

    /**
     * Checks every input of an embedding-row call before its first step: every row against the accepted widths
     * ({@link #acceptsRowWidth}) and the positions {@code pos0 ..} against the position table, so a rejected call
     * has not written memory.
     *
     * @throws IllegalArgumentException when a row is null or another width than {@code n_embd} or
     *                                 {@code nEmbdInp}, the row array is empty, or a position is out of range
     */
    public void checkRowsInput(float[][] rows, int pos0) {
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0) {
            throw new IllegalArgumentException("no embedding rows to evaluate");
        }
        checkRowWidths(rows, 0, rows.length);
        long end = (long) pos0 + rows.length;
        if (pos0 < 0 || end > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("positions [" + pos0 + ", " + end + ") out of range");
        }
        if (posTable != null) {
            checkRow("position", (int) (end - 1) + posOffset, posTable);
        }
    }

    /**
     * Embeds precomputed rows {@code rows[off .. off + n)} at positions {@code pos0 ..} into the residual stream:
     * each row replaces the token-table lookup (vLLM {@code prompt_embeds}, llama.cpp's embedding batch), then
     * the type and position additions and the input norm run as for token rows (llama.cpp builds them after
     * {@code build_inp_embd}), so a row holding the token table's own values produces the same model input as
     * that token. The LoRA embedding deltas and the embedding scale belong to the lookup and do not run. An
     * {@code n_embd} row is zero-padded to {@code nEmbdInp}; a full-width row keeps its tail.
     * {@link CpuScratch#TOKEN_IDS} holds {@code -1} per embedding row.
     *
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, ubatch]}, a range is outside its
     *                                 array, a row is null or another width than {@code n_embd} or
     *                                 {@code nEmbdInp}, or a position is out of range
     */
    public void embedRows(float[][] rows, int off, int n, int pos0, float[] x) {
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(x, "x");
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("step of " + n + " embedding rows outside [1, "
                    + scratch.ubatch() + "]");
        }
        if (off < 0 || (long) off + n > rows.length) {
            throw new IllegalArgumentException("row range [" + off + ", " + ((long) off + n) + ") outside "
                    + rows.length + " rows");
        }
        if (pos0 < 0 || (long) pos0 + n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("positions [" + pos0 + ", " + ((long) pos0 + n) + ") out of range");
        }
        if ((long) n * streamWidth > x.length) {
            throw new IllegalArgumentException("residual stream holds " + x.length + " floats, need "
                    + (long) n * streamWidth);
        }
        checkRowWidths(rows, off, n);
        RopePositions.fillStep(hp.mropeTextPositions(), pos0, n, scratch.ints(CpuScratch.POS));
        writeRows(rows, off, n, null, pos0, x);
    }

    /**
     * The shared body of the embedding-row steps over validated rows and filled positions: the null token per
     * row, each row copied into {@link CpuScratch#INPUT_EMBD} and completed by the input-row tail at its position
     * ({@code rowPos[t]}, or {@code pos0 + t} when {@code rowPos} is null), and the stream copy.
     */
    private void writeRows(float[][] rows, int off, int n, int[] rowPos, int pos0, float[] x) {
        int[] ids = scratch.ints(CpuScratch.TOKEN_IDS);
        for (int t = 0; t < n; t++) {
            ids[t] = -1;
            ids[n + t] = 0;
        }
        float[] inp = scratch.slot(CpuScratch.INPUT_EMBD);
        CpuParallel.parallelFor(n, TOKEN_GRAIN, (from, to) -> {
            for (int t = from; t < to; t++) {
                float[] row = rows[off + t];
                System.arraycopy(row, 0, inp, t * nEmbdInp, row.length);
                finishInputRow(t, rowPos != null ? rowPos[t] : pos0 + t, 0, inp, row.length);
            }
        });
        copyToStream(n, x);
    }

    /** Whether a caller-provided embedding row of {@code width} values fits: {@code n_embd} or {@code nEmbdInp}. */
    public boolean acceptsRowWidth(int width) {
        return width == nEmbd || width == nEmbdInp;
    }

    /** Every row of the range is a non-null array of {@code n_embd} or {@code nEmbdInp} values. */
    private void checkRowWidths(float[][] rows, int off, int n) {
        for (int t = 0; t < n; t++) {
            float[] row = rows[off + t];
            if (row == null || !acceptsRowWidth(row.length)) {
                throw new IllegalArgumentException("embedding row " + (off + t) + (row == null ? " is null"
                        : " holds " + row.length + " values, expected n_embd " + nEmbd
                        + (nEmbdInp != nEmbd ? " or the model input width " + nEmbdInp : "")));
            }
        }
    }

    /** Copies the built {@link CpuScratch#INPUT_EMBD} rows over the {@code streams} of the residual stream. */
    private void copyToStream(int n, float[] x) {
        float[] inp = scratch.slot(CpuScratch.INPUT_EMBD);
        CpuParallel.parallelFor(n, TOKEN_GRAIN, (from, to) -> {
            for (int t = from; t < to; t++) {
                int src = t * nEmbdInp;
                for (int h = 0; h < streams; h++) {
                    System.arraycopy(inp, src, x, (t * streams + h) * nEmbd, nEmbd);
                }
            }
        });
    }

    /** Builds row {@code t} of {@link CpuScratch#INPUT_EMBD} from a token-table lookup. */
    private void embedToken(int t, int pos, int[] stepIds, int type, float[] inp) {
        int base = t * nEmbdInp;
        long token = checkRow("token id", stepIds[t], tokenTable);
        if (inputProjection == null) {
            weights.embedRow(tokenTable, token, inp, base);
            if (!embdDeltas.isEmpty()) {
                embdDeltas.applyRow(t, stepIds, t, inp, base);
            }
            scaleRow(inp, base);
        } else {
            int lookup = t * rowWidth;
            weights.embedRow(tokenTable, token, rowTemp, lookup);
            if (!embdDeltas.isEmpty()) {
                embdDeltas.applyRow(t, stepIds, t, rowTemp, lookup);
            }
            scaleRow(rowTemp, lookup);
            matVec.matmul(inputProjection, rowTemp, lookup, inp, base, 1);
        }
        finishInputRow(t, pos, type, inp, nEmbd);
    }

    /** {@code embdScale} on the looked-up token row (its {@link #tokenWidth} columns). */
    private void scaleRow(float[] row, int off) {
        float scale = embd.embdScale();
        if (scale == 1.0f) {
            return;
        }
        for (int i = 0; i < tokenWidth; i++) {
            row[off + i] *= scale;
        }
    }

    /**
     * Completes INPUT_EMBD row {@code t} once its first {@code filled} values are written (at least the
     * {@code n_embd} the trunk reads): the type and position additions and the input norm on the first
     * {@code n_embd} values, and the zero padding of {@code [filled, nEmbdInp)}.
     */
    private void finishInputRow(int t, int pos, int type, float[] inp, int filled) {
        int base = t * nEmbdInp;
        int tmp = t * rowWidth;
        if (typeTable != null) {
            weights.embedRow(typeTable, checkRow("token type", type, typeTable), rowTemp, tmp);
            for (int i = 0; i < nEmbd; i++) {
                inp[base + i] += rowTemp[tmp + i];
            }
        }
        if (posTable != null) {
            weights.embedRow(posTable, checkRow("position", pos + posOffset, posTable), rowTemp, tmp);
            for (int i = 0; i < nEmbd; i++) {
                inp[base + i] = rowTemp[tmp + i] + inp[base + i];
            }
        }
        if (normWeight != null) {
            NormMath.rowNorm(NormKind.LAYER, hp.normEps(), inp, base, inp, base, nEmbd, normWeight, 0, normBias, 0);
        }
        if (embd.inputRmsNorm()) {
            NormMath.rowNorm(NormKind.RMS, hp.normRmsEps(), inp, base, inp, base, nEmbd, inputRmsNormWeight, 0,
                    null, 0);
        }
        for (int i = filled; i < nEmbdInp; i++) {
            inp[base + i] = 0f;
        }
    }

    private static long checkRow(String what, int row, CpuTensorRef table) {
        if (row < 0 || row >= table.rows()) {
            throw new IllegalArgumentException(what + " " + row + " outside [0, " + table.rows() + ")");
        }
        return row;
    }

    @Override
    public String toString() {
        return "CpuEmbeddingStage{n_embd=" + nEmbd + ", inp=" + nEmbdInp + ", streams=" + streams
                + (typeTable != null ? ", types" : "") + (posTable != null ? ", pos" : "")
                + (normWeight != null ? ", norm" : "") + '}';
    }
}
