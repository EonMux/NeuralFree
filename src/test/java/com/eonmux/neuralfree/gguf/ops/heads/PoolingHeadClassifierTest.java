package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
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
 * The classifier chain over every token (vLLM's token classification / process reward) and over one pooled row
 * (sequence classification): {@code cls} -> activation -> LayerNorm (+ bias) -> {@code cls.output}, with the
 * per-token rows equal to the same math applied to each row, and the refusals that keep a token classifier's
 * scores away from the embedding outputs.
 */
public class PoolingHeadClassifierTest {

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

    private ModelSpec spec(PoolingKind pooling) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.BERT)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_IN).normEps(EPS).pooling(pooling).causal(false)
                .classifierActivation(ActOp.RELU);
        HeadSpec head = HeadSpec.builder().pooling(pooling)
                .classifier(catalog.get(TensorNames.weight(TensorNames.CLS)),
                        catalog.get(TensorNames.weight(TensorNames.CLS_OUT)),
                        catalog.get(TensorNames.weight(TensorNames.CLS_NORM)))
                .classifierScope(pooling == PoolingKind.NONE ? ClassifierScope.TOKEN : ClassifierScope.SEQUENCE)
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

    @Test
    public void tokenClassifierAppliesTheChainToEveryRow() throws IOException {
        open();
        ModelSpec spec = spec(PoolingKind.NONE);
        assertTrue(EncodeGate.hasTokenClassifier(spec));
        assertFalse(EncodeGate.hasSequenceClassifier(spec));
        PoolingHead head = PoolingHead.of(spec, weights);
        assertTrue(head.tokenClassifier());
        assertEquals(N_OUT, head.outputWidth());

        float[][] scores = head.classifyTokens(ROWS);
        assertEquals(ROWS.length, scores.length);
        for (int t = 0; t < ROWS.length; t++) {
            assertArrayEquals("token " + t, chain(ROWS[t]), scores[t], TOL);
        }
    }

    @Test
    public void pooledClassifierAppliesTheChainToTheClsRow() throws IOException {
        open();
        ModelSpec spec = spec(PoolingKind.CLS);
        assertTrue(EncodeGate.hasSequenceClassifier(spec));
        PoolingHead head = PoolingHead.of(spec, weights);
        assertTrue(head.sequenceClassifier());
        assertArrayEquals(chain(ROWS[0]), head.pool(ROWS), TOL);
        ClassifierOutput output = head.classify(ROWS);
        assertArrayEquals(chain(ROWS[0]), output.scores(), TOL);
        assertEquals(chain(ROWS[0])[0], head.rerank(ROWS), 0f);
    }

    @Test
    public void theTwoScopesRefuseEachOthersPooledOutputs() throws IOException {
        open();
        PoolingHead token = PoolingHead.of(spec(PoolingKind.NONE), weights);
        assertThrows(IllegalStateException.class, () -> token.pool(ROWS));
        assertThrows(IllegalStateException.class, () -> token.poolTokens(ROWS));
        assertThrows(IllegalStateException.class, () -> token.classify(ROWS));

        PoolingHead pooled = PoolingHead.of(spec(PoolingKind.CLS), weights);
        assertThrows(IllegalStateException.class, () -> pooled.poolTokens(ROWS));
        float[][] rows = pooled.classifyTokens(ROWS);
        assertEquals("a sequence classifier also answers token_classify", ROWS.length, rows.length);
        for (int t = 0; t < ROWS.length; t++) {
            assertArrayEquals(chain(ROWS[t]), rows[t], TOL);
        }
    }

    @Test
    public void embeddingOutputsRefuseClassifierHeads() throws IOException {
        open();
        ModelSpec spec = spec(PoolingKind.NONE);
        assertThrows(UnsupportedOperationException.class, () -> EncodeGate.checkEncode(spec));
        assertThrows(UnsupportedOperationException.class, () -> EncodeGate.checkEncodeTokens(spec));
        assertThrows(UnsupportedOperationException.class, () -> EncodeGate.checkRerank(spec));
        EncodeGate.checkTokenClassify(spec);

        ModelSpec sequence = spec(PoolingKind.CLS);
        assertThrows(UnsupportedOperationException.class, () -> EncodeGate.checkEncode(sequence));
        EncodeGate.checkTokenClassify(sequence);
        EncodeGate.checkRerank(sequence);
    }
}
