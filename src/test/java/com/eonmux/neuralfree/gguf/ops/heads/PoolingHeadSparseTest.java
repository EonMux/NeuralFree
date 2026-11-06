package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The sparse embedding ({@code classifier.sparse}, vLLM's SPLADESparsePooler): the per-token classifier scores
 * through {@code log1p(relu(x))}, reduced by max or sum over the rows, without the leading CLS and trailing SEP
 * rows; the metadata gate keeps the reduced vector out of the per-token outputs.
 */
public class PoolingHeadSparseTest {

    private static final int N_IN = 4;
    private static final int N_HIDDEN = 3;
    private static final int N_OUT = 2;
    private static final float EPS = 1e-5f;
    private static final float TOL = 1e-5f;

    /** {@code cls.weight} {N_IN, N_HIDDEN}: column {@code o} holds the inputs of hidden unit {@code o}. */
    private static final float[] W1 = {
            0.5f, -1.0f, 0.25f, 2.0f,
            1.0f, 0.0f, -0.5f, 0.75f,
            -0.25f, 1.5f, 1.0f, -1.0f};
    private static final float[] B1 = {0.1f, -0.2f, 0.3f};
    private static final float[] NORM_W = {1.5f, 0.5f, 2.0f};
    private static final float[] NORM_B = {0.25f, -0.5f, 0.1f};
    /** {@code cls.output.weight} {N_HIDDEN, N_OUT}. */
    private static final float[] W2 = {
            1.0f, -0.5f, 0.25f,
            -1.0f, 2.0f, 0.5f};
    private static final float[] B2 = {0.05f, -0.15f};
    private static final float[][] ROWS = {
            {1.0f, 2.0f, -1.0f, 0.5f},
            {0.0f, -1.0f, 3.0f, 1.0f},
            {2.0f, 0.5f, 0.5f, -2.0f}};

    private Path path;
    private GGUFFileParser parser;
    private CpuWeights weights;
    private TensorCatalog catalog;

    private void open() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.BERT.ggufName());
        w.tensorF32(TensorNames.weight(TensorNames.CLS), new long[]{N_IN, N_HIDDEN}, W1);
        w.tensorF32(TensorNames.bias(TensorNames.CLS), new long[]{N_HIDDEN}, B1);
        w.tensorF32(TensorNames.weight(TensorNames.CLS_NORM), new long[]{N_HIDDEN}, NORM_W);
        w.tensorF32(TensorNames.bias(TensorNames.CLS_NORM), new long[]{N_HIDDEN}, NORM_B);
        w.tensorF32(TensorNames.weight(TensorNames.CLS_OUT), new long[]{N_HIDDEN, N_OUT}, W2);
        w.tensorF32(TensorNames.bias(TensorNames.CLS_OUT), new long[]{N_OUT}, B2);
        path = w.writeTempFile();
        parser = new GGUFFileParser(path.toString());
        weights = CpuWeights.open(parser);
        catalog = TensorCatalog.of(parser);
    }

    @After
    public void close() throws IOException {
        try {
            if (weights != null) {
                weights.close();
            }
            if (parser != null) {
                parser.close();
            }
        } finally {
            if (path != null) {
                Files.deleteIfExists(path);
            }
        }
    }

    private ModelSpec spec(String sparse) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.BERT)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_IN).normEps(EPS).pooling(PoolingKind.NONE).causal(false)
                .classifierActivation(ActOp.RELU)
                .classifierSparse(sparse).sparseClsToken(101).sparseSepToken(102);
        HeadSpec head = HeadSpec.builder().pooling(PoolingKind.NONE)
                .classifier(catalog.get(TensorNames.weight(TensorNames.CLS)),
                        catalog.get(TensorNames.weight(TensorNames.CLS_OUT)),
                        catalog.get(TensorNames.weight(TensorNames.CLS_NORM)))
                .classifierScope(ClassifierScope.TOKEN)
                .build();
        return ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(null, N_IN).build())
                .head(head)
                .build();
    }

    /** relu(W1 x + b1), LayerNorm with weight and bias, W2 + b2, in double precision. */
    private static float[] chain(float[] x) {
        float[] h = new float[N_HIDDEN];
        for (int o = 0; o < N_HIDDEN; o++) {
            double acc = B1[o];
            for (int i = 0; i < N_IN; i++) {
                acc += (double) W1[o * N_IN + i] * x[i];
            }
            h[o] = (float) Math.max(0.0, acc);
        }
        double mean = 0;
        for (float v : h) {
            mean += v;
        }
        mean /= N_HIDDEN;
        double var = 0;
        for (float v : h) {
            var += (v - mean) * (v - mean);
        }
        var /= N_HIDDEN;
        double inv = 1.0 / Math.sqrt(var + EPS);
        float[] y = new float[N_OUT];
        for (int o = 0; o < N_OUT; o++) {
            double acc = B2[o];
            for (int h1 = 0; h1 < N_HIDDEN; h1++) {
                double norm = ((h[h1] - mean) * inv) * NORM_W[h1] + NORM_B[h1];
                acc += (double) W2[o * N_HIDDEN + h1] * norm;
            }
            y[o] = (float) acc;
        }
        return y;
    }

    /** {@code log1p(max(v, 0))} of the chain over every row, reduced by max or sum over rows {@code [from, to)}. */
    private static float[] sparse(float[][] rows, int from, int to, boolean sum) {
        float[] out = new float[N_OUT];
        for (int t = from; t < to; t++) {
            float[] scores = chain(rows[t]);
            for (int c = 0; c < N_OUT; c++) {
                float w = (float) Math.log1p(Math.max(scores[c], 0.0));
                out[c] = sum ? out[c] + w : Math.max(out[c], w);
            }
        }
        return out;
    }

    @Test
    public void maxReductionDropsTheBoundaryRows() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec("log1p_relu_max"), weights);
        assertTrue(head.sparsePooling() != null);
        assertEquals(N_OUT, head.outputWidth());
        // the first row carries the CLS id and the last the SEP id; both drop before the reduction
        assertArrayEquals(sparse(ROWS, 1, 2, false), head.sparseEmbed(ROWS, new int[]{101, 5, 102}), TOL);
    }

    @Test
    public void sumReductionSumsEveryKeptRow() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec("log1p_relu_sum"), weights);
        assertArrayEquals(sparse(ROWS, 0, ROWS.length, true),
                head.sparseEmbed(ROWS, new int[]{7, 8, 9}), TOL);
    }

    @Test
    public void onlyTheFirstAndLastRowsDrop() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec("log1p_relu_max"), weights);
        float[][] four = {ROWS[0], ROWS[1], ROWS[2], ROWS[1]};
        // boundary ids count only at the ends; the middle rows with the same ids stay
        assertArrayEquals(sparse(four, 1, four.length - 1, false),
                head.sparseEmbed(four, new int[]{101, 102, 101, 102}), TOL);
    }

    @Test
    public void withoutTheMetadataTheSparseOutputRefuses() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec(null), weights);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> head.sparseEmbed(ROWS, new int[]{101, 102, 103}));
        assertTrue(e.getMessage(), e.getMessage().contains("classifier.sparse"));
    }

    @Test
    public void allBoundaryRowsRefuse() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec("log1p_relu_max"), weights);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> head.sparseEmbed(new float[][]{ROWS[0], ROWS[1]}, new int[]{101, 102}));
        assertTrue(e.getMessage(), e.getMessage().contains("every token row drops"));
    }

    @Test
    public void mismatchedTokenIdsRefuse() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec("log1p_relu_max"), weights);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> head.sparseEmbed(ROWS, new int[]{101, 102}));
        assertTrue(e.getMessage(), e.getMessage().contains("token ids"));
    }

    @Test
    public void theSparseModelAnswersEmbedAndPlainTokenRows() throws IOException {
        open();
        ModelSpec spec = spec("log1p_relu_max");
        EncodeGate.checkEncode(spec);
        assertTrue("the plain token rows are the model's token_embed output", EncodeGate.embedsPerToken(spec));
        EncodeGate.checkEncodeTokens(spec);
        assertThrows(UnsupportedOperationException.class, () -> EncodeGate.checkTokenClassify(spec));
        assertEquals("log1p_relu_max", EncodeGate.sparsePooling(spec));
        PoolingHead head = PoolingHead.of(spec, weights);
        float[][] rows = head.poolTokens(ROWS);
        assertEquals(ROWS.length, rows.length);
        for (int t = 0; t < rows.length; t++) {
            assertArrayEquals("the token-embed rows are the embedding chain's, not the classifier's",
                    ROWS[t], rows[t], TOL);
        }
    }
}
