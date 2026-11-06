package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * Embedding pooling and the sentence-transformers dense projections over the final hidden rows
 * ({@code ForwardBackend.hiddenRows}: after the output norm and the embedding projection, rows of
 * {@link HeadSpec#hiddenWidth(int)} floats), shared by the GPU and CPU orchestrators so every backend pools with the
 * same semantics:
 * <ul>
 *   <li>embedding output (qwen35 {@code cls.output}, {@link HeadSpec#embeddingOutput()}): the embedding rows are
 *       {@code W h + b} of {@link HeadSpec#embeddingWidth(int)} floats (models/qwen35.cpp:222-229). It is applied
 *       to every row for NONE, to the selected row for CLS and LAST, and to the mean for MEAN, which equals the mean
 *       of the projected rows because the projection is affine;</li>
 *   <li>pooling (llama-graph.cpp:4090-4183 build_pooling): NONE keeps every row; MEAN averages the rows; CLS
 *       takes the first row and LAST the last one; RANK pools by {@link CpuPooling#rowPooling} (MEAN when
 *       {@code classifier.pooling_type} is MEAN, else LAST for causal and CLS for non-causal attention) and then
 *       runs the {@link ClassifierHead};</li>
 *   <li>classifier head ({@code cls} / {@code cls.norm} / {@code cls.output}): RANK and the pooled types run it
 *       over the pooled row (a sequence classifier), and NONE runs it over every token (vLLM's
 *       {@code token_classify});</li>
 *   <li>dense out (build_dense_out, llama-graph.cpp:4065-4087, applied after pooling, llama-model.cpp:3338-3342): when
 *       {@code dense_2}, {@code dense_2.bias} or {@code dense_3} exists, {@code x = dense_3 (dense_2 x + b2)} on the
 *       pooled vector, or on every row for NONE;</li>
 *   <li>L2-normalized embedding rows ({@code pooling.normalize}, vLLM's PoolerNormalize): every embedding row is
 *       unit-normalized after the projections, per token for NONE and on the pooled vector otherwise;</li>
 *   <li>the direct token classification ({@code classifier.token_direct}, vLLM's {@code for_seq_cls} over a
 *       prediction-head classifier): per-token classification applies only {@code cls.output}, skipping the
 *       {@code cls} / {@code cls.norm} head, which the pooled classification still runs;</li>
 *   <li>the sparse embedding ({@code classifier.sparse}, vLLM's {@code SPLADESparsePooler}): the per-token
 *       classifier scores go through {@code log1p(relu(x))} and reduce by max or sum over the rows, without the
 *       leading CLS and trailing SEP rows.</li>
 * </ul>
 * The forward backends read back the hidden rows once per call; the head then runs on the host over the mapped
 * weights. Immutable; every call allocates its result.
 */
public final class PoolingHead {

    private final PoolingKind pooling;
    private final PoolingKind rowPooling;
    private final int hiddenWidth;
    private final int embeddingWidth;
    private final CpuMatVec mv;
    private final CpuTensorRef embeddingOutput;
    private final float[] embeddingOutputB;
    private final CpuTensorRef dense2;
    private final float[] dense2B;
    private final CpuTensorRef dense3;
    private final ClassifierHead classifier;
    private final CpuTensorRef colbert;
    private final float[] colbertB;
    private final int colbertWidth;
    private final CpuTensorRef sparseProjection;
    private final float[] sparseB;
    private final int sparseWidth;
    private final int tokenBos;
    private final int tokenEos;
    private final boolean normalize;
    private final boolean tokenDirect;
    private final String sparse;
    private final int sparseClsToken;
    private final int sparseSepToken;
    private final int outWidth;
    private final int tokenWidth;

    private PoolingHead(ModelSpec spec, CpuWeights w) {
        HeadSpec head = spec.head();
        int nEmbd = spec.hparams().nEmbd();
        this.pooling = head.pooling();
        this.rowPooling = CpuPooling.rowPooling(pooling, spec.hparams().poolingCls(), spec.hparams().causal());
        this.hiddenWidth = head.hiddenWidth(nEmbd);
        this.embeddingWidth = head.embeddingWidth(nEmbd);
        this.normalize = spec.hparams().poolingNormalize();
        this.tokenDirect = spec.hparams().classifierTokenDirect();
        this.sparse = spec.hparams().classifierSparse();
        this.sparseClsToken = spec.hparams().sparseClsToken();
        this.sparseSepToken = spec.hparams().sparseSepToken();
        this.mv = w.matVec();
        this.embeddingOutput = head.embeddingOutput() != null ? w.get(head.embeddingOutput()) : null;
        this.embeddingOutputB = head.embeddingOutputBias() != null ? w.floats(head.embeddingOutputBias()) : null;
        if (embeddingOutput != null && (embeddingOutput.cols() != hiddenWidth
                || embeddingOutput.rows() != embeddingWidth)) {
            throw new IllegalArgumentException("embedding output " + embeddingOutput + " is " + embeddingOutput.cols()
                    + " x " + embeddingOutput.rows() + " (in x out), expected " + hiddenWidth + " x " + embeddingWidth);
        }
        if (embeddingOutputB != null && embeddingOutputB.length != embeddingWidth) {
            throw new IllegalArgumentException("embedding output bias has " + embeddingOutputB.length
                    + " floats, expected " + embeddingWidth);
        }
        this.classifier = runsClassifier(spec) ? ClassifierHead.of(spec, w) : null;
        this.tokenBos = spec.hparams().classifierBosToken();
        this.tokenEos = spec.hparams().classifierEosToken();
        this.colbert = head.colbertOutput() != null ? w.get(head.colbertOutput()) : null;
        this.colbertB = head.colbertOutputBias() != null ? w.floats(head.colbertOutputBias()) : null;
        this.colbertWidth = this.colbert == null ? 0 : checkTokenHead(this.colbert, colbertB, hiddenWidth,
                "cls.colbert");
        this.sparseProjection = head.sparseOutput() != null ? w.get(head.sparseOutput()) : null;
        this.sparseB = head.sparseOutputBias() != null ? w.floats(head.sparseOutputBias()) : null;
        this.sparseWidth = this.sparseProjection == null ? 0
                : checkTokenHead(this.sparseProjection, sparseB, hiddenWidth, "cls.sparse");
        this.dense2 = w.getOrNull(TensorNames.weight(TensorNames.DENSE_2_OUT));
        this.dense2B = w.floatsOrNull(TensorNames.bias(TensorNames.DENSE_2_OUT));
        this.dense3 = w.getOrNull(TensorNames.weight(TensorNames.DENSE_3_OUT));
        int width = classifier != null ? classifier.outputs() : embeddingWidth;
        if (dense2 != null && sparse == null) {
            width = checkDense(dense2, width);
        }
        if (dense2B != null && sparse == null && dense2B.length != width) {
            throw new IllegalArgumentException("dense_2.bias has " + dense2B.length + " floats, the pooled vector "
                    + width);
        }
        if (dense3 != null && sparse == null) {
            width = checkDense(dense3, width);
        }
        // the token-embed rows of a sparse model pass the same dense projections as the embedding rows, before
        // the classifier scores; their width is the embedding chain's, not the classifier's
        this.tokenWidth = sparse == null ? width : denseChainWidth(dense2, dense3);
        if (tokenDirect && classifier != null && !classifier.hasOutput()) {
            throw new IllegalArgumentException("classifier.token_direct needs a cls.output projection; the head"
                    + " stores none");
        }
        this.outWidth = width;
    }

    /** The width of the dense chain over {@link #embeddingWidth} rows, for a sparse model's token-embed output. */
    private int denseChainWidth(CpuTensorRef dense2W, CpuTensorRef dense3W) {
        int w = embeddingWidth;
        if (dense2W != null) {
            w = checkDense(dense2W, w);
        }
        if (dense3W != null) {
            w = checkDense(dense3W, w);
        }
        return w;
    }

    /**
     * The pooling head of {@code spec} over the tensors of {@code weights}.
     *
     * @throws IllegalArgumentException when a head tensor does not take the previous stage's width
     */
    public static PoolingHead of(ModelSpec spec, CpuWeights weights) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(weights, "weights");
        return new PoolingHead(spec, weights);
    }

    /**
     * Whether the classifier chain runs for {@code spec}: a scoped head runs over every token ({@code token}) or
     * over the pooled row ({@code sequence}); without a scope llama.cpp's legacy classifier runs only for RANK
     * pooling, so a RANK model overridden to another pooling type drops its classifier and qwen35's
     * {@code cls.output} embedding rows stay embeddings. A decision model scores through {@link DecisionScorer},
     * unless it reads its fixed question over the pooled classifier output (llama.cpp's minicpm-ocr-jev), which
     * runs the chain over the pooled row like a sequence classifier.
     */
    private static boolean runsClassifier(ModelSpec spec) {
        if (EncodeGate.hasDecisionClassifier(spec)) {
            return true;
        }
        if (DecisionScorer.present(spec)) {
            return false;
        }
        ClassifierScope scope = spec.head().classifierScope();
        if (scope == ClassifierScope.TOKEN) {
            return spec.head().pooling() == PoolingKind.NONE;
        }
        if (scope == ClassifierScope.SEQUENCE) {
            return spec.head().pooling() != PoolingKind.NONE;
        }
        return spec.head().pooling() == PoolingKind.RANK;
    }

    private static int checkDense(CpuTensorRef w, int width) {
        if (w.cols() != width) {
            throw new IllegalArgumentException("dense projection " + w + " takes " + w.cols() + " inputs, the pooled"
                    + " vector has " + width);
        }
        return HeadsBlocks.toInt(w.rows(), w);
    }

    /**
     * The output width of a per-token head projection over {@code width} input floats.
     *
     * @throws IllegalArgumentException when the projection takes another input width or its bias has another size
     */
    private static int checkTokenHead(CpuTensorRef w, float[] bias, int width, String what) {
        if (w.cols() != width) {
            throw new IllegalArgumentException(what + " projection " + w + " takes " + w.cols() + " inputs, the"
                    + " hidden row has " + width);
        }
        int out = HeadsBlocks.toInt(w.rows(), w);
        if (bias != null && bias.length != out) {
            throw new IllegalArgumentException(what + " bias has " + bias.length + " floats, the projection outputs "
                    + out);
        }
        return out;
    }

    /** The model's pooling type. */
    public PoolingKind pooling() {
        return pooling;
    }

    /** The row pooling applied before the classifier ({@link PoolingKind#RANK} resolved). */
    public PoolingKind rowPooling() {
        return rowPooling;
    }

    /** Floats per hidden row the head reads ({@link HeadSpec#hiddenWidth(int)}). */
    public int hiddenWidth() {
        return hiddenWidth;
    }

    /** Width of each output vector: the embedding width, the classifier outputs, or the last dense projection's rows. */
    public int outputWidth() {
        return outWidth;
    }

    /** The classifier chain (per token or pooled), or null. */
    public ClassifierHead classifier() {
        return classifier;
    }

    /** True when NONE pooling runs the classifier chain over every token (vLLM's {@code token_classify}). */
    public boolean tokenClassifier() {
        return classifier != null && pooling == PoolingKind.NONE;
    }

    /** True when a pooled type runs the classifier chain over the pooled row (a sequence classifier). */
    public boolean sequenceClassifier() {
        return classifier != null && pooling != PoolingKind.NONE;
    }

    /**
     * The per-token classifier scores of every hidden row: {@code [nTokens][classifier().outputs()]}. A token
     * classifier's own output; a sequence classifier answers vLLM's {@code token_classify} task with the same
     * chain over each row. A multi-output model (BGE-M3's {@code cls.sparse}) projects every row through its
     * {@code sparse_linear} and applies the ReLU vLLM's token-classify pooler holds.
     *
     * @throws IllegalStateException for a model without a classifier head
     */
    public float[][] classifyTokens(float[][] hiddenRows) {
        return classifyTokens(hiddenRows, null);
    }

    /**
     * {@link #classifyTokens(float[][])} with the prompt token ids: the leading and trailing rows whose ids are
     * {@code classifier.bos_token} and {@code classifier.eos_token} drop (vLLM's {@code BOSEOSFilter}); a null
     * array keeps every row.
     *
     * @throws IllegalStateException    for a model without a classifier head
     * @throws IllegalArgumentException when {@code tokens} does not match the row count
     */
    public float[][] classifyTokens(float[][] hiddenRows, int[] tokens) {
        if (classifier == null && sparseProjection == null) {
            throw new IllegalStateException("per-token classification needs a classifier head; this head is " + this);
        }
        int n = hiddenRows == null ? 0 : hiddenRows.length;
        float[] embd = embed(HeadsBlocks.pack(hiddenRows, hiddenWidth), n);
        int width;
        float[] y;
        if (sparseProjection != null) {
            y = HeadsBlocks.linear(mv, sparseProjection, sparseB, embd, n);
            HeadsBlocks.activate(ActOp.RELU, y, y.length);
            width = sparseWidth;
        } else {
            float[] scores = tokenDirect
                    ? classifier.applyOutputRows(embd, n, true)
                    : classifier.applyRows(embd, n);
            y = dense(scores, n);
            width = outWidth;
        }
        int start = boundaryStart(n, tokens, tokenBos);
        int end = boundaryEnd(n, tokens, tokenEos, start);
        float[][] rows = new float[end - start][];
        for (int t = start; t < end; t++) {
            rows[t - start] = new float[width];
            System.arraycopy(y, t * width, rows[t - start], 0, width);
        }
        return rows;
    }

    /** The first kept row: 1 when the leading row carries {@code bosId}, else 0. */
    private static int boundaryStart(int n, int[] tokens, int bosId) {
        checkTokens(n, tokens);
        return n > 0 && tokens != null && bosId >= 0 && tokens[0] == bosId ? 1 : 0;
    }

    /** The end (exclusive) of the kept rows: {@code n - 1} when the trailing row carries {@code eosId}. */
    private static int boundaryEnd(int n, int[] tokens, int eosId, int start) {
        if (n > start && tokens != null && eosId >= 0 && tokens[n - 1] == eosId) {
            return n - 1;
        }
        return n;
    }

    private static void checkTokens(int n, int[] tokens) {
        if (tokens != null && tokens.length != n) {
            throw new IllegalArgumentException(tokens.length + " token ids for " + n + " hidden rows");
        }
    }

    /**
     * The pooled output of one sequence: the pooled embedding (MEAN / CLS / LAST), the classifier scores of the
     * pooled row (RANK or any pooled classifier), through the dense projections.
     *
     * @throws IllegalStateException    for NONE pooling, which has one output per row ({@link #poolTokens} and
     *                                  {@link #classifyTokens})
     * @throws IllegalArgumentException when a row does not hold {@link #hiddenWidth()} floats
     */
    public float[] pool(float[][] hiddenRows) {
        return pool(hiddenRows, true);
    }

    /**
     * {@link #pool(float[][])} with the classifier's final probability activation applied or skipped (vLLM's
     * {@code use_activation}); {@code false} answers the raw head output, and an embedding model pools the same
     * either way.
     *
     * @throws IllegalStateException    for NONE pooling, which has one output per row ({@link #poolTokens})
     * @throws IllegalArgumentException when a row does not hold {@link #hiddenWidth()} floats
     */
    public float[] pool(float[][] hiddenRows, boolean activation) {
        if (pooling == PoolingKind.NONE) {
            throw new IllegalStateException("NONE pooling has one output per token; use poolTokens or"
                    + " classifyTokens");
        }
        int n = hiddenRows == null ? 0 : hiddenRows.length;
        float[] x = HeadsBlocks.pack(hiddenRows, hiddenWidth);
        float[] pooled = new float[hiddenWidth];
        CpuPooling.pool(rowPooling, n, hiddenWidth, x, 0, hiddenWidth, pooled, 0);
        float[] embd = embed(pooled, 1);
        float[] out = classifier != null ? classifier.apply(embd, 0, activation) : embd;
        float[] y = dense(out, 1);
        if (normalize && classifier == null) {
            normalizeRow(y, 0, outWidth);
        }
        return y;
    }

    /**
     * Per-token embedding rows of NONE pooling: every embedding row through the dense projections. A sparse model
     * answers these plain rows beside its sparse {@code embed} output (vLLM's {@code token_embed} on
     * {@code BertSpladeSparseEmbeddingModel}); a multi-output model projects them through {@code cls.colbert}
     * (BGE-M3's {@code colbert_linear}) and L2-normalizes them.
     *
     * @throws IllegalStateException for a pooled type without a {@code cls.colbert} projection ({@link #pool}), and
     *                               for a legacy token classifier, whose per-token rows are classifier scores
     *                               ({@link #classifyTokens})
     */
    public float[][] poolTokens(float[][] hiddenRows) {
        return poolTokens(hiddenRows, null);
    }

    /**
     * {@link #poolTokens(float[][])} with the prompt token ids: the leading row whose id is
     * {@code classifier.bos_token} drops. vLLM wraps this task in a {@code BOSEOSFilter} that leaves the EOS id
     * disabled for the per-token embedding rows, so a trailing EOS row stays. A null array keeps every row.
     *
     * @throws IllegalStateException    for a pooled type, and for a token classifier without a separate per-token
     *                                  embedding projection
     * @throws IllegalArgumentException when {@code tokens} does not match the row count
     */
    public float[][] poolTokens(float[][] hiddenRows, int[] tokens) {
        if (pooling != PoolingKind.NONE && colbert == null) {
            throw new IllegalStateException(pooling + " pooling has one output per sequence; use pool");
        }
        if (tokenClassifier() && colbert == null && sparse == null) {
            throw new IllegalStateException("this model is a token classifier; its per-token rows are classifier"
                    + " scores, not embeddings; use classifyTokens");
        }
        int n = hiddenRows == null ? 0 : hiddenRows.length;
        float[] embd = embed(HeadsBlocks.pack(hiddenRows, hiddenWidth), n);
        float[] y;
        int width;
        if (colbert != null) {
            y = HeadsBlocks.linear(mv, colbert, colbertB, embd, n);
            width = colbertWidth;
        } else {
            y = dense(embd, n);
            width = tokenWidth;
        }
        if (normalize) {
            for (int t = 0; t < n; t++) {
                normalizeRow(y, t * width, width);
            }
        }
        int start = boundaryStart(n, tokens, tokenBos);
        float[][] rows = new float[n - start][width];
        for (int t = start; t < n; t++) {
            System.arraycopy(y, t * width, rows[t - start], 0, width);
        }
        return rows;
    }

    /** The width of a per-token embedding row: the {@code cls.colbert} projection's, else the dense chain's. */
    public int tokenEmbeddingWidth() {
        return colbert != null ? colbertWidth : tokenWidth;
    }

    /** The number of scores each per-token classification row holds ({@code cls.sparse}'s rows). */
    public int tokenScoreWidth() {
        return sparseProjection != null ? sparseWidth : outWidth;
    }

    /** True when the model stores the BGE-M3-style per-token embedding projection ({@code cls.colbert}). */
    public boolean hasTokenEmbeddingProjection() {
        return colbert != null;
    }

    /** True when the model stores the BGE-M3-style per-token score projection ({@code cls.sparse}). */
    public boolean hasTokenScoreProjection() {
        return sparseProjection != null;
    }

    /**
     * vLLM's combined {@code embed&token_classify} output (BgeM3Pooler): the pooled embedding vector followed by
     * the flattened per-token classifier scores, one value per kept token row.
     *
     * @throws IllegalStateException    for a model without a pooled embedding or a per-token score projection
     * @throws IllegalArgumentException when {@code tokens} does not match the row count
     */
    public float[] embedAndTokenClassify(float[][] hiddenRows, int[] tokens) {
        if (pooling == PoolingKind.NONE) {
            throw new IllegalStateException("the combined output needs a pooled embedding; this head pools per token");
        }
        if (sparseProjection == null) {
            throw new IllegalStateException("the combined output needs the " + TensorNames.CLS_SPARSE
                    + " per-token score projection; this head has none");
        }
        float[] pooled = pool(hiddenRows, false);
        float[][] rows = classifyTokens(hiddenRows, tokens);
        int width = sparseWidth;
        float[] out = new float[pooled.length + rows.length * width];
        System.arraycopy(pooled, 0, out, 0, pooled.length);
        for (int t = 0; t < rows.length; t++) {
            System.arraycopy(rows[t], 0, out, pooled.length + t * width, width);
        }
        return out;
    }

    /**
     * The sparse embedding of one sequence (vLLM's {@code SPLADESparsePooler}): the classifier scores of every
     * token, {@code log1p(relu(x))} per score, reduced over the rows by {@code classifier.sparse} ({@code max} or
     * {@code sum}); the leading row whose token id is {@code classifier.sparse_cls_token} and the trailing row
     * whose id is {@code classifier.sparse_sep_token} drop first (vLLM's {@code remove_cls_sep}). A vector of
     * {@link #classifier()}{@code .outputs()} entries, answered by the {@code embed} task.
     *
     * @throws IllegalStateException when the model has no sparse pooling, or every row drops
     * @throws IllegalArgumentException when {@code tokens} does not match the row count
     */
    public float[] sparseEmbed(float[][] hiddenRows, int[] tokens) {
        if (sparse == null) {
            throw new IllegalStateException("sparse embedding needs the " + GgufKeys.CLASSIFIER_SPARSE
                    + " metadata; this head is " + this);
        }
        int n = hiddenRows == null ? 0 : hiddenRows.length;
        if (tokens != null && tokens.length != n) {
            throw new IllegalArgumentException(tokens.length + " token ids for " + n + " hidden rows");
        }
        int start = n > 0 && tokens != null && tokens.length == n && tokens[0] == sparseClsToken ? 1 : 0;
        int end = n > start && tokens != null && tokens.length == n && tokens[n - 1] == sparseSepToken ? n - 1 : n;
        if (end <= start) {
            throw new IllegalStateException("every token row drops as the sparse pooling's cls (" + sparseClsToken
                    + ") or sep (" + sparseSepToken + ") id; the input has " + n + " rows");
        }
        float[] embd = embed(HeadsBlocks.pack(hiddenRows, hiddenWidth), n);
        float[] scores = classifier.applyRows(embd, n);
        int width = classifier.outputs();
        boolean sum = "log1p_relu_sum".equals(sparse);
        float[] out = new float[width];
        for (int t = start; t < end; t++) {
            for (int c = 0; c < width; c++) {
                float x = scores[t * width + c];
                float w = (float) Math.log1p(Math.max(x, 0.0f));
                out[c] = sum ? out[c] + w : Math.max(out[c], w);
            }
        }
        return out;
    }

    /** The sparse pooling over the per-token classifier scores ({@code classifier.sparse}), or null. */
    public String sparsePooling() {
        return sparse;
    }

    /** L2-normalizes one row in place; a zero row stays zero ({@code ggml_l2_norm}). */
    private static void normalizeRow(float[] x, int off, int width) {
        CpuOps.norm(NormKind.L2_CLAMP, 1e-12f, x, off, x, off, 1, width, null, null);
    }

    /**
     * The classifier output of a pooled classifier ({@link #sequenceClassifier()}) with its labels.
     *
     * @throws IllegalStateException when the model has no pooled classifier, or classifies per token
     */
    public ClassifierOutput classify(float[][] hiddenRows) {
        return classify(hiddenRows, true);
    }

    /**
     * {@link #classify(float[][])} with the head's final probability activation applied or skipped (vLLM's
     * {@code use_activation}); {@code false} answers the raw scores, which are not normalized.
     *
     * @throws IllegalStateException when the model's pooling is not RANK
     */
    public ClassifierOutput classify(float[][] hiddenRows, boolean activation) {
        if (classifier == null) {
            throw new IllegalStateException("classification needs a classifier head; the model has none");
        }
        if (tokenClassifier()) {
            throw new IllegalStateException("this model classifies per token; use classifyTokens");
        }
        return ClassifierOutput.of(pool(hiddenRows, activation), classifier.labels(),
                classifier.normalized() && activation);
    }

    /**
     * The relevance score of a rerank input ({@link ClassifierHead#rerankInput}): the first classifier output, as
     * the llama.cpp server reports it.
     *
     * @throws IllegalStateException when the model has no pooled classifier, or classifies per token
     */
    public float rerank(float[][] hiddenRows) {
        return rerank(hiddenRows, true);
    }

    /**
     * {@link #rerank(float[][])} with the head's final probability activation applied or skipped (vLLM's
     * {@code use_activation}); {@code false} answers the raw head score.
     *
     * @throws IllegalStateException when the model's pooling is not RANK
     */
    public float rerank(float[][] hiddenRows, boolean activation) {
        return classify(hiddenRows, activation).score(0);
    }

    /** The embedding rows of {@code nRows} packed hidden rows: the embedding output when the model has one. */
    private float[] embed(float[] x, int nRows) {
        if (embeddingOutput == null) {
            return x;
        }
        return HeadsBlocks.linear(mv, embeddingOutput, embeddingOutputB, x, nRows);
    }

    private float[] dense(float[] x, int nRows) {
        float[] y = x;
        if (dense2 != null) {
            y = HeadsBlocks.linear(mv, dense2, null, y, nRows);
        }
        if (dense2B != null) {
            HeadsBlocks.addBias(y, nRows, dense2B.length, dense2B, "dense_2");
        }
        if (dense3 != null) {
            y = HeadsBlocks.linear(mv, dense3, null, y, nRows);
        }
        return y;
    }

    @Override
    public String toString() {
        return "PoolingHead{" + pooling + (pooling == PoolingKind.RANK ? "(" + rowPooling + ")" : "")
                + ", in=" + hiddenWidth + (embeddingOutput != null ? ", embd=" + embeddingWidth : "")
                + ", out=" + outWidth + '}';
    }
}
