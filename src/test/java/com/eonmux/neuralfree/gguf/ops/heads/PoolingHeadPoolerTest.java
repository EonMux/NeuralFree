package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolerAct;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
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
 * The pooler-level behaviors the GGUF metadata keys carry: the L2-normalized embedding rows ({@code
 * pooling.normalize}, vLLM's PoolerNormalize), the pooler classifier activation ({@code
 * classifier.pooler_activation}, vLLM's problem_type / sentence-transformers resolution) and the direct token
 * classification ({@code classifier.token_direct}, vLLM's {@code for_seq_cls} over a ModernBert classifier, which
 * applies only the final projection per token).
 */
public class PoolingHeadPoolerTest {

    private static final int N_IN = 4;
    private static final int N_OUT = 2;
    private static final float EPS = 1e-5f;
    private static final float TOL = 1e-5f;

    /** {@code cls.weight} {N_IN, N_IN}: row {@code o} holds the inputs of hidden unit {@code o}. */
    private static final float[] W1 = {
            0.5f, -1.0f, 0.25f, 2.0f,
            1.0f, 0.0f, -0.5f, 0.75f,
            -0.25f, 1.5f, 1.0f, -1.0f,
            0.75f, -0.5f, 1.25f, 0.0f};
    private static final float[] B1 = {0.1f, -0.2f, 0.3f, -0.15f};
    private static final float[] NORM_W = {1.5f, 0.5f, 2.0f, 1.0f};
    private static final float[] NORM_B = {0.25f, -0.5f, 0.1f, -0.25f};
    /**
     * {@code cls.output.weight} {N_IN, N_OUT}: the prediction head keeps the hidden width, so the projection
     * follows both the chain and, per {@code classifier.token_direct}, the raw rows.
     */
    private static final float[] W2 = {
            0.5f, -1.5f, 0.75f, 0.25f,
            -0.25f, 1.0f, -0.5f, 1.25f};
    private static final float[] B2 = {0.05f, -0.15f};
    /** {@code dense_2.weight} {N_IN, D_OUT}: row {@code o} holds the inputs of output unit {@code o}. */
    private static final float[] DENSE = {
            1.0f, 0.0f, -0.5f, 1.5f,
            0.25f, -1.0f, 2.0f, 0.5f,
            -0.75f, 0.5f, 1.0f, -1.25f};
    private static final int D_OUT = 3;
    private static final float[][] ROWS = {
            {1.0f, 2.0f, -1.0f, 0.5f},
            {0.0f, -1.0f, 3.0f, 1.0f},
            {2.0f, 0.5f, 0.5f, -2.0f}};

    private Path path;
    private GGUFFileParser parser;
    private CpuWeights weights;
    private TensorCatalog catalog;

    /** The pooled embedding model carries the {@code dense_2} projection, the classifier model the chain. */
    private void open(boolean classifier) throws IOException {
        open(classifier, true);
    }

    /**
     * Opens the fixture with or without the {@code cls.output} tensor: the head reads the projection from the
     * file, so a {@code token_direct} model without it has nothing to apply.
     */
    private void open(boolean classifier, boolean clsOutput) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.BERT.ggufName());
        if (!classifier) {
            w.tensorF32(TensorNames.weight(TensorNames.DENSE_2_OUT), new long[]{N_IN, D_OUT}, DENSE);
        } else {
            w.tensorF32(TensorNames.weight(TensorNames.CLS), new long[]{N_IN, N_IN}, W1);
            w.tensorF32(TensorNames.bias(TensorNames.CLS), new long[]{N_IN}, B1);
            w.tensorF32(TensorNames.weight(TensorNames.CLS_NORM), new long[]{N_IN}, NORM_W);
            w.tensorF32(TensorNames.bias(TensorNames.CLS_NORM), new long[]{N_IN}, NORM_B);
            if (clsOutput) {
                w.tensorF32(TensorNames.weight(TensorNames.CLS_OUT), new long[]{N_IN, N_OUT}, W2);
                w.tensorF32(TensorNames.bias(TensorNames.CLS_OUT), new long[]{N_OUT}, B2);
            }
        }
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

    /** An embedding model over the dense_2 projection, pooled or per token, with or without the L2 norm. */
    private ModelSpec embeddingSpec(PoolingKind pooling, boolean normalize) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.BERT)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_IN).normEps(EPS).pooling(pooling).causal(false)
                .poolingNormalize(normalize);
        HeadSpec head = HeadSpec.builder().pooling(pooling)
                .dense(catalog.get(TensorNames.weight(TensorNames.DENSE_2_OUT)), null)
                .build();
        return model(hb, head);
    }

    /** A pooled sequence classifier over the full chain, with the pooler activation set or not. */
    private ModelSpec classifySpec(PoolerAct act) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.BERT)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_IN).normEps(EPS).pooling(PoolingKind.CLS).causal(false)
                .poolerActivation(act);
        return model(hb, classifierHead(ClassifierScope.SEQUENCE));
    }

    /** A token classifier over every row, through the whole chain or only the {@code cls.output} projection. */
    private ModelSpec tokenSpec(boolean direct) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.BERT)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_IN).normEps(EPS).pooling(PoolingKind.NONE).causal(false)
                .classifierTokenDirect(direct);
        return model(hb, classifierHead(ClassifierScope.TOKEN));
    }

    private HeadSpec classifierHead(ClassifierScope scope) {
        return HeadSpec.builder()
                .pooling(scope == ClassifierScope.TOKEN ? PoolingKind.NONE : PoolingKind.CLS)
                .classifier(catalog.get(TensorNames.weight(TensorNames.CLS)),
                        catalog.get(TensorNames.weight(TensorNames.CLS_OUT)),
                        catalog.get(TensorNames.weight(TensorNames.CLS_NORM)))
                .classifierScope(scope)
                .build();
    }

    private ModelSpec model(ModelHparamsBuilder hb, HeadSpec head) {
        return ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(null, N_IN).build())
                .head(head)
                .build();
    }

    /** tanh(W1 x + b1), LayerNorm with weight and bias, W2 + b2, in double precision. */
    private static float[] chain(float[] x) {
        float[] h = new float[N_IN];
        for (int o = 0; o < N_IN; o++) {
            double acc = B1[o];
            for (int i = 0; i < N_IN; i++) {
                acc += (double) W1[o * N_IN + i] * x[i];
            }
            h[o] = (float) Math.tanh(acc);
        }
        double mean = 0;
        for (float v : h) {
            mean += v;
        }
        mean /= N_IN;
        double var = 0;
        for (float v : h) {
            var += (v - mean) * (v - mean);
        }
        var /= N_IN;
        double inv = 1.0 / Math.sqrt(var + EPS);
        float[] y = new float[N_OUT];
        for (int o = 0; o < N_OUT; o++) {
            double acc = B2[o];
            for (int i = 0; i < N_IN; i++) {
                double norm = ((h[i] - mean) * inv) * NORM_W[i] + NORM_B[i];
                acc += (double) W2[o * N_IN + i] * norm;
            }
            y[o] = (float) acc;
        }
        return y;
    }

    /** {@code cls.output} over the raw embedding rows only, the direct token classification's projection. */
    private static float[] output(float[] x) {
        float[] y = new float[N_OUT];
        for (int o = 0; o < N_OUT; o++) {
            double acc = B2[o];
            for (int i = 0; i < N_IN; i++) {
                acc += (double) W2[o * N_IN + i] * x[i];
            }
            y[o] = (float) acc;
        }
        return y;
    }

    private static float[] dense(float[] x) {
        float[] y = new float[D_OUT];
        for (int o = 0; o < D_OUT; o++) {
            double acc = 0;
            for (int i = 0; i < N_IN; i++) {
                acc += (double) DENSE[o * N_IN + i] * x[i];
            }
            y[o] = (float) acc;
        }
        return y;
    }

    private static void l2(float[] x) {
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        double inv = 1.0 / Math.sqrt(Math.max(sum, 1e-12));
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (x[i] * inv);
        }
    }

    private static float norm(float[] x) {
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        return (float) Math.sqrt(sum);
    }

    private static float[] mean() {
        float[] m = new float[N_IN];
        for (float[] row : ROWS) {
            for (int i = 0; i < N_IN; i++) {
                m[i] += row[i] / ROWS.length;
            }
        }
        return m;
    }

    private static float[] softmax(float[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : x) {
            max = Math.max(max, v);
        }
        double sum = 0;
        double[] e = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            e[i] = Math.exp(x[i] - max);
            sum += e[i];
        }
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = (float) (e[i] / sum);
        }
        return y;
    }

    @Test
    public void normalizeAppliesToThePooledVectorAndEveryTokenRow() throws IOException {
        open(false);
        PoolingHead head = PoolingHead.of(embeddingSpec(PoolingKind.MEAN, true), weights);
        float[] pooled = head.pool(ROWS);
        assertEquals(1.0f, norm(pooled), 1e-4f);
        float[] expected = dense(mean());
        l2(expected);
        assertArrayEquals(expected, pooled, TOL);

        PoolingHead perToken = PoolingHead.of(embeddingSpec(PoolingKind.NONE, true), weights);
        float[][] tokenRows = perToken.poolTokens(ROWS);
        assertEquals(ROWS.length, tokenRows.length);
        for (int t = 0; t < ROWS.length; t++) {
            assertEquals("token " + t, 1.0f, norm(tokenRows[t]), 1e-4f);
            float[] row = dense(ROWS[t]);
            l2(row);
            assertArrayEquals(row, tokenRows[t], TOL);
        }
    }

    @Test
    public void withoutTheNormalizeFlagTheRowsStayRaw() throws IOException {
        open(false);
        PoolingHead head = PoolingHead.of(embeddingSpec(PoolingKind.MEAN, false), weights);
        assertArrayEquals(dense(mean()), head.pool(ROWS), TOL);
        PoolingHead perToken = PoolingHead.of(embeddingSpec(PoolingKind.NONE, false), weights);
        float[][] tokenRows = perToken.poolTokens(ROWS);
        for (int t = 0; t < ROWS.length; t++) {
            assertArrayEquals(dense(ROWS[t]), tokenRows[t], TOL);
        }
    }

    @Test
    public void softmaxPoolerActivationNormalizesTheScores() throws IOException {
        open(true);
        PoolingHead head = PoolingHead.of(classifySpec(PoolerAct.SOFTMAX), weights);
        assertTrue(head.classifier().normalized());
        assertArrayEquals(softmax(chain(ROWS[0])), head.pool(ROWS), TOL);
        assertTrue(head.classify(ROWS).normalized());
        assertArrayEquals(chain(ROWS[0]), head.classify(ROWS, false).scores(), TOL);
    }

    @Test
    public void sigmoidPoolerActivationSquashesEveryScore() throws IOException {
        open(true);
        PoolingHead head = PoolingHead.of(classifySpec(PoolerAct.SIGMOID), weights);
        assertFalse(head.classifier().normalized());
        float[] scores = chain(ROWS[0]);
        float[] expected = new float[N_OUT];
        for (int i = 0; i < N_OUT; i++) {
            expected[i] = (float) (1.0 / (1.0 + Math.exp(-scores[i])));
        }
        assertArrayEquals(expected, head.pool(ROWS), TOL);
    }

    @Test
    public void tanhPoolerActivationSquashesEveryScore() throws IOException {
        open(true);
        PoolingHead head = PoolingHead.of(classifySpec(PoolerAct.TANH), weights);
        assertFalse(head.classifier().normalized());
        float[] scores = chain(ROWS[0]);
        float[] expected = new float[N_OUT];
        for (int i = 0; i < N_OUT; i++) {
            expected[i] = (float) Math.tanh(scores[i]);
        }
        assertArrayEquals(expected, head.pool(ROWS), TOL);
    }

    @Test
    public void identityPoolerActivationKeepsTheRawScores() throws IOException {
        open(true);
        PoolingHead head = PoolingHead.of(classifySpec(PoolerAct.IDENTITY), weights);
        assertFalse(head.classifier().normalized());
        assertArrayEquals(chain(ROWS[0]), head.pool(ROWS), TOL);
    }

    @Test
    public void directTokenClassificationAppliesOnlyTheOutputProjection() throws IOException {
        open(true);
        PoolingHead direct = PoolingHead.of(tokenSpec(true), weights);
        float[][] rows = direct.classifyTokens(ROWS);
        for (int t = 0; t < ROWS.length; t++) {
            assertArrayEquals(output(ROWS[t]), rows[t], TOL);
        }
        // without the flag the same head runs the whole chain, whose scores differ
        PoolingHead full = PoolingHead.of(tokenSpec(false), weights);
        float[][] chained = full.classifyTokens(ROWS);
        for (int t = 0; t < ROWS.length; t++) {
            assertArrayEquals(chain(ROWS[t]), chained[t], TOL);
        }
    }

    @Test
    public void directClassificationNeedsTheOutputProjection() throws IOException {
        open(true, false);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> PoolingHead.of(tokenSpec(true), weights));
        assertTrue(e.getMessage(), e.getMessage().contains("classifier.token_direct"));
    }
}
