package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
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
import static org.junit.Assert.assertThrows;

/**
 * The host pooling head with an embedding output (qwen35 {@code cls.output}, models/qwen35.cpp:222-229): the pooled
 * vector equals llama.cpp's pooling of the projected rows {@code W h + b} for MEAN, CLS and NONE, the RANK classifier
 * applies {@code cls.output} once more to the pooled embedding as build_pooling does, and a classifier whose input
 * differs from the embedding width is rejected.
 */
public class PoolingHeadTest {

    private static final int N_EMBD = 4;
    private static final int N_OUT = 3;
    private static final float TOL = 1e-5f;
    /** {@code cls.output.weight} {N_EMBD, N_OUT}: row {@code o} holds the inputs of output {@code o}. */
    private static final float[] W = {
            0.5f, -1.0f, 0.25f, 2.0f,
            1.0f, 0.0f, -0.5f, 0.75f,
            -0.25f, 1.5f, 1.0f, -1.0f};
    private static final float[] B = {0.1f, -0.2f, 0.3f};
    private static final float[][] ROWS = {
            {1.0f, 2.0f, -1.0f, 0.5f},
            {0.0f, -1.0f, 3.0f, 1.0f},
            {2.0f, 0.5f, 0.5f, -2.0f}};

    private Path path;
    private GGUFFileParser parser;
    private CpuWeights weights;
    private TensorCatalog catalog;

    private void open(int nOut) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.QWEN35.ggufName());
        float[] weight = new float[N_EMBD * nOut];
        for (int i = 0; i < weight.length; i++) {
            weight[i] = W[i % W.length];
        }
        float[] bias = new float[nOut];
        for (int i = 0; i < nOut; i++) {
            bias[i] = B[i % B.length];
        }
        w.tensorF32(TensorNames.weight(TensorNames.CLS_OUT), new long[]{N_EMBD, nOut}, weight);
        w.tensorF32(TensorNames.bias(TensorNames.CLS_OUT), new long[]{nOut}, bias);
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

    private ModelSpec spec(PoolingKind pooling, int nOut) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN35)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_EMBD).nEmbdOutImpl(nOut).normRmsEps(1e-6f).pooling(pooling).causal(true);
        HeadSpec head = HeadSpec.builder().pooling(pooling)
                .classifier(null, catalog.get(TensorNames.weight(TensorNames.CLS_OUT)), null)
                .embeddingOutput(catalog.get(TensorNames.weight(TensorNames.CLS_OUT)),
                        catalog.get(TensorNames.bias(TensorNames.CLS_OUT)))
                .build();
        return ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(head)
                .build();
    }

    /** {@code W x + b} of one row in double precision. */
    private static float[] project(float[] x) {
        float[] y = new float[N_OUT];
        for (int o = 0; o < N_OUT; o++) {
            double acc = B[o];
            for (int i = 0; i < N_EMBD; i++) {
                acc += (double) W[o * N_EMBD + i] * x[i];
            }
            y[o] = (float) acc;
        }
        return y;
    }

    @Test
    public void meanPoolsTheProjectedRows() throws IOException {
        open(N_OUT);
        PoolingHead head = PoolingHead.of(spec(PoolingKind.MEAN, N_OUT), weights);
        assertEquals(N_EMBD, head.hiddenWidth());
        assertEquals(N_OUT, head.outputWidth());

        float[] expected = new float[N_OUT];
        for (float[] row : ROWS) {
            float[] y = project(row);
            for (int o = 0; o < N_OUT; o++) {
                expected[o] += y[o] / ROWS.length;
            }
        }
        assertArrayEquals(expected, head.pool(ROWS), TOL);
    }

    @Test
    public void lastAndNonePoolingProjectEachRow() throws IOException {
        open(N_OUT);
        assertArrayEquals(project(ROWS[2]), PoolingHead.of(spec(PoolingKind.LAST, N_OUT), weights).pool(ROWS), TOL);

        float[][] tokens = PoolingHead.of(spec(PoolingKind.NONE, N_OUT), weights).poolTokens(ROWS);
        assertEquals(ROWS.length, tokens.length);
        for (int t = 0; t < ROWS.length; t++) {
            assertArrayEquals(project(ROWS[t]), tokens[t], TOL);
        }
    }

    @Test
    public void rankAppliesTheOutputAgainToThePooledEmbedding() throws IOException {
        open(N_EMBD);
        PoolingHead head = PoolingHead.of(spec(PoolingKind.RANK, N_EMBD), weights);
        assertEquals(PoolingKind.LAST, head.rowPooling());

        float[] w = new float[N_EMBD * N_EMBD];
        for (int i = 0; i < w.length; i++) {
            w[i] = W[i % W.length];
        }
        float[] once = affine(w, ROWS[2]);
        assertArrayEquals(affine(w, once), head.classify(ROWS).scores(), TOL);
    }

    @Test
    public void rankWithAnEmbeddingWidthOtherThanNEmbdIsRejected() throws IOException {
        open(N_OUT);
        assertThrows(IllegalArgumentException.class, () -> PoolingHead.of(spec(PoolingKind.RANK, N_OUT), weights));
    }

    /** {@code W x + b} for a square {@code N_EMBD} weight with the cyclic bias. */
    private static float[] affine(float[] w, float[] x) {
        float[] y = new float[N_EMBD];
        for (int o = 0; o < N_EMBD; o++) {
            double acc = B[o % B.length];
            for (int i = 0; i < N_EMBD; i++) {
                acc += (double) w[o * N_EMBD + i] * x[i];
            }
            y[o] = (float) acc;
        }
        return y;
    }
}
