package com.eonmux.neuralfree.gguf.cpu.head;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
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
 *   <li>{@code + token_types[type]} (type 0 when no types are given, as llama.cpp's "sentence A") and
 *       {@code + position_embd[pos]} when the tables exist;</li>
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
 * <p>Not thread-safe; one forward pass at a time. A step allocates nothing.</p>
 */
public final class CpuEmbeddingStage {

    private static final String TEMP_ROW = "CpuEmbeddingStage.row";
    /** Tokens per parallel task of the lookup. */
    private static final int TOKEN_GRAIN = 4;

    private final ModelHparams hp;
    private final CpuWeights weights;
    private final CpuScratch scratch;
    private final EmbeddingSpec embd;
    private final CpuTensorRef tokenTable;
    private final CpuTensorRef typeTable;
    private final CpuTensorRef posTable;
    private final float[] normWeight;
    private final float[] normBias;
    private final int nEmbd;
    private final int nEmbdInp;
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
        this.tokenTable = table(embd.tokenEmbd(), "token_embd");
        this.typeTable = table(embd.tokenTypes(), "token_types");
        this.posTable = table(embd.posEmbd(), "position_embd");
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
        this.rowTemp = scratch.temp(TEMP_ROW, Math.multiplyExact(scratch.ubatch(), nEmbd));
        this.embdDeltas = tokenTable == null ? null : ctx.projectionDeltas().serveEmbedding(
                TensorNames.weight(TensorNames.TOKEN_EMBD), tokenTable.rows(), nEmbd, null);
    }

    private CpuTensorRef table(TensorRef ref, String what) {
        if (ref == null) {
            return null;
        }
        CpuTensorRef t = weights.get(ref);
        if (t.cols() != nEmbd) {
            throw new IllegalArgumentException(what + " rows hold " + t.cols() + " values, expected n_embd " + nEmbd);
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
            checkRow("position", (int) (end - 1), posTable);
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
        CpuParallel.parallelFor(n, TOKEN_GRAIN, (from, to) -> {
            for (int t = from; t < to; t++) {
                int src = t * nEmbdInp;
                for (int h = 0; h < streams; h++) {
                    System.arraycopy(inp, src, x, (t * streams + h) * nEmbd, nEmbd);
                }
            }
        });
    }

    private void writeStepInputs(int[] tokens, int[] tokenTypes, int off, int n, int pos0) {
        int[] ids = scratch.ints(CpuScratch.TOKEN_IDS);
        for (int t = 0; t < n; t++) {
            ids[t] = tokens[off + t];
            ids[n + t] = tokenTypes != null ? tokenTypes[off + t] : 0;
        }
        RopePositions.fillStep(hp.mropeTextPositions(), pos0, n, scratch.ints(CpuScratch.POS));
    }

    /** Builds row {@code t} of {@link CpuScratch#INPUT_EMBD}. */
    private void embedToken(int t, int pos, int[] stepIds, int type, float[] inp) {
        int base = t * nEmbdInp;
        weights.embedRow(tokenTable, checkRow("token id", stepIds[t], tokenTable), inp, base);
        if (!embdDeltas.isEmpty()) {
            embdDeltas.apply(1, stepIds, t, inp, base, nEmbdInp);
        }
        float scale = embd.embdScale();
        if (scale != 1.0f) {
            for (int i = 0; i < nEmbd; i++) {
                inp[base + i] *= scale;
            }
        }
        int tmp = t * nEmbd;
        if (typeTable != null) {
            weights.embedRow(typeTable, checkRow("token type", type, typeTable), rowTemp, tmp);
            for (int i = 0; i < nEmbd; i++) {
                inp[base + i] += rowTemp[tmp + i];
            }
        }
        if (posTable != null) {
            weights.embedRow(posTable, checkRow("position", pos, posTable), rowTemp, tmp);
            for (int i = 0; i < nEmbd; i++) {
                inp[base + i] = rowTemp[tmp + i] + inp[base + i];
            }
        }
        if (normWeight != null) {
            NormMath.rowNorm(NormKind.LAYER, hp.normEps(), inp, base, inp, base, nEmbd, normWeight, 0, normBias, 0);
        }
        if (embd.inputRmsNorm()) {
            NormMath.rowNorm(NormKind.RMS, hp.normRmsEps(), inp, base, inp, base, nEmbd, null, 0, null, 0);
        }
        for (int i = nEmbd; i < nEmbdInp; i++) {
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
