package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
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
 *   <li>dense out (build_dense_out, llama-graph.cpp:4065-4087, applied after pooling, llama-model.cpp:3338-3342): when
 *       {@code dense_2}, {@code dense_2.bias} or {@code dense_3} exists, {@code x = dense_3 (dense_2 x + b2)} on the
 *       pooled vector, or on every row for NONE.</li>
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
    private final int outWidth;

    private PoolingHead(ModelSpec spec, CpuWeights w) {
        HeadSpec head = spec.head();
        int nEmbd = spec.hparams().nEmbd();
        this.pooling = head.pooling();
        this.rowPooling = CpuPooling.rowPooling(pooling, spec.hparams().poolingCls(), spec.hparams().causal());
        this.hiddenWidth = head.hiddenWidth(nEmbd);
        this.embeddingWidth = head.embeddingWidth(nEmbd);
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
        this.classifier = pooling == PoolingKind.RANK ? ClassifierHead.of(spec, w) : null;
        this.dense2 = w.getOrNull(TensorNames.weight(TensorNames.DENSE_2_OUT));
        this.dense2B = w.floatsOrNull(TensorNames.bias(TensorNames.DENSE_2_OUT));
        this.dense3 = w.getOrNull(TensorNames.weight(TensorNames.DENSE_3_OUT));
        int width = classifier != null ? classifier.outputs() : embeddingWidth;
        if (dense2 != null) {
            width = checkDense(dense2, width);
        }
        if (dense2B != null && dense2B.length != width) {
            throw new IllegalArgumentException("dense_2.bias has " + dense2B.length + " floats, the pooled vector "
                    + width);
        }
        if (dense3 != null) {
            width = checkDense(dense3, width);
        }
        this.outWidth = width;
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

    private static int checkDense(CpuTensorRef w, int width) {
        if (w.cols() != width) {
            throw new IllegalArgumentException("dense projection " + w + " takes " + w.cols() + " inputs, the pooled"
                    + " vector has " + width);
        }
        return HeadsBlocks.toInt(w.rows(), w);
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

    /** The ranking classifier of RANK pooling, or null. */
    public ClassifierHead classifier() {
        return classifier;
    }

    /**
     * The pooled output of one sequence: the pooled embedding (MEAN / CLS / LAST) or the classifier scores (RANK),
     * through the dense projections.
     *
     * @throws IllegalStateException    for NONE pooling, which has one output per row ({@link #poolTokens})
     * @throws IllegalArgumentException when a row does not hold {@link #hiddenWidth()} floats
     */
    public float[] pool(float[][] hiddenRows) {
        if (pooling == PoolingKind.NONE) {
            throw new IllegalStateException("NONE pooling has one embedding per token; use poolTokens");
        }
        int n = hiddenRows == null ? 0 : hiddenRows.length;
        float[] x = HeadsBlocks.pack(hiddenRows, hiddenWidth);
        float[] pooled = new float[hiddenWidth];
        CpuPooling.pool(rowPooling, n, hiddenWidth, x, 0, hiddenWidth, pooled, 0);
        float[] embd = embed(pooled, 1);
        float[] out = classifier != null ? classifier.apply(embd, 0) : embd;
        return dense(out, 1);
    }

    /**
     * Per-token outputs of NONE pooling: every embedding row through the dense projections.
     *
     * @throws IllegalStateException for a pooled type ({@link #pool})
     */
    public float[][] poolTokens(float[][] hiddenRows) {
        if (pooling != PoolingKind.NONE) {
            throw new IllegalStateException(pooling + " pooling has one output per sequence; use pool");
        }
        int n = hiddenRows == null ? 0 : hiddenRows.length;
        float[] y = dense(embed(HeadsBlocks.pack(hiddenRows, hiddenWidth), n), n);
        float[][] rows = new float[n][outWidth];
        for (int t = 0; t < n; t++) {
            System.arraycopy(y, t * outWidth, rows[t], 0, outWidth);
        }
        return rows;
    }

    /**
     * The classifier output of RANK pooling with its labels.
     *
     * @throws IllegalStateException when the model's pooling is not RANK
     */
    public ClassifierOutput classify(float[][] hiddenRows) {
        if (classifier == null) {
            throw new IllegalStateException("classification needs RANK pooling, the model pools by " + pooling);
        }
        return ClassifierOutput.of(pool(hiddenRows), classifier.labels(), classifier.normalized());
    }

    /**
     * The relevance score of a rerank input ({@link ClassifierHead#rerankInput}): the first classifier output, as
     * the llama.cpp server reports it.
     *
     * @throws IllegalStateException when the model's pooling is not RANK
     */
    public float rerank(float[][] hiddenRows) {
        return classify(hiddenRows).score(0);
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
