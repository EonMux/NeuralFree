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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The BGE-M3 multi-output head (V/model_executor/models/roberta.py:187-269, layers/pooler/special.py:225-260): the
 * CLS-pooled dense {@code embed}, the {@code cls.colbert} per-token rows ({@code token_embed}, L2-normalized), the
 * {@code cls.sparse} ReLU per-token scores ({@code token_classify}), the leading-BOS drop of the token-embed rows and
 * the BOS/EOS drop of the classifier rows, and vLLM's combined
 * {@code embed&token_classify} concatenation. The expected values are the scalar transcription of vLLM's pooler
 * over the same weights.
 */
public class PoolingHeadMultiOutputTest {

    private static final int N_IN = 4;
    private static final int N_COLBERT = 3;
    private static final float EPS = 1e-5f;
    private static final float TOL = 1e-5f;
    private static final int BOS = 2;
    private static final int EOS = 3;

    /** {@code cls.colbert.weight} {N_IN, N_COLBERT}: column {@code o} holds the inputs of output {@code o}. */
    private static final float[] COLBERT_W = {
            0.5f, -1.0f, 0.25f, 2.0f,
            1.0f, 0.0f, -0.5f, 0.75f,
            -0.25f, 1.5f, 1.0f, -1.0f};
    private static final float[] COLBERT_B = {0.1f, -0.2f, 0.3f};
    /** {@code cls.sparse.weight} {N_IN, 1}. */
    private static final float[] SPARSE_W = {0.5f, -1.5f, 2.0f, 0.25f};
    private static final float[] SPARSE_B = {0.05f};
    private static final float[][] ROWS = {
            {1.0f, 2.0f, -1.0f, 0.5f},
            {0.0f, -1.0f, 3.0f, 1.0f},
            {2.0f, 0.5f, 0.5f, -2.0f},
            {-0.5f, 1.0f, 0.0f, 2.0f}};

    private Path path;
    private GGUFFileParser parser;
    private CpuWeights weights;
    private TensorCatalog catalog;

    private void open() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.BERT.ggufName());
        w.tensorF32(TensorNames.weight(TensorNames.CLS_COLBERT), new long[]{N_IN, N_COLBERT}, COLBERT_W);
        w.tensorF32(TensorNames.bias(TensorNames.CLS_COLBERT), new long[]{N_COLBERT}, COLBERT_B);
        w.tensorF32(TensorNames.weight(TensorNames.CLS_SPARSE), new long[]{N_IN, 1}, SPARSE_W);
        w.tensorF32(TensorNames.bias(TensorNames.CLS_SPARSE), new long[]{1}, SPARSE_B);
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

    private ModelSpec spec() {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.BERT)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_IN).normEps(EPS).pooling(PoolingKind.CLS).causal(false)
                .poolingNormalize(true).classifierBosToken(BOS).classifierEosToken(EOS);
        HeadSpec head = HeadSpec.builder().pooling(PoolingKind.CLS)
                .tokenHeads(catalog.get(TensorNames.weight(TensorNames.CLS_COLBERT)),
                        catalog.get(TensorNames.bias(TensorNames.CLS_COLBERT)),
                        catalog.get(TensorNames.weight(TensorNames.CLS_SPARSE)),
                        catalog.get(TensorNames.bias(TensorNames.CLS_SPARSE)))
                .build();
        return ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(null, N_IN).build())
                .head(head)
                .build();
    }

    /** vLLM's colbert_linear row plus the per-row L2 normalization. */
    private static float[] colbert(float[] x) {
        float[] y = new float[N_COLBERT];
        double sum = 0;
        for (int o = 0; o < N_COLBERT; o++) {
            double acc = COLBERT_B[o];
            for (int i = 0; i < N_IN; i++) {
                acc += (double) COLBERT_W[o * N_IN + i] * x[i];
            }
            y[o] = (float) acc;
            sum += (double) y[o] * y[o];
        }
        double norm = Math.sqrt(sum);
        for (int o = 0; o < N_COLBERT; o++) {
            y[o] /= (float) norm;
        }
        return y;
    }

    /** vLLM's sparse_linear followed by the token pooler's ReLU. */
    private static float sparse(float[] x) {
        double acc = SPARSE_B[0];
        for (int i = 0; i < N_IN; i++) {
            acc += (double) SPARSE_W[i] * x[i];
        }
        return (float) Math.max(acc, 0.0);
    }

    /** The CLS row, colbert-projected? No: the pooled dense embedding is the CLS hidden row, L2-normalized. */
    private static float[] pooledDense(float[][] rows) {
        float[] y = rows[0].clone();
        double sum = 0;
        for (float v : y) {
            sum += (double) v * v;
        }
        double norm = Math.sqrt(sum);
        for (int i = 0; i < y.length; i++) {
            y[i] /= (float) norm;
        }
        return y;
    }

    @Test
    public void tokenEmbedAnswersColbertRowsWithoutTheLeadingBosRow() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec(), weights);
        assertTrue(head.hasTokenEmbeddingProjection());
        assertEquals(N_COLBERT, head.tokenEmbeddingWidth());
        // vLLM's token-embed BOSEOSFilter leaves the EOS id disabled, so the trailing row stays
        float[][] rows = head.poolTokens(ROWS, new int[]{BOS, 7, 8, EOS});
        assertEquals(3, rows.length);
        assertArrayEquals(colbert(ROWS[1]), rows[0], TOL);
        assertArrayEquals(colbert(ROWS[2]), rows[1], TOL);
        assertArrayEquals(colbert(ROWS[3]), rows[2], TOL);
    }

    @Test
    public void tokenClassifyAnswersTheSparseReLUWithoutTheBoundaryRows() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec(), weights);
        assertTrue(head.hasTokenScoreProjection());
        assertEquals(1, head.tokenScoreWidth());
        float[][] rows = head.classifyTokens(ROWS, new int[]{BOS, 7, 8, EOS});
        assertEquals(2, rows.length);
        assertEquals(sparse(ROWS[1]), rows[0][0], TOL);
        assertEquals(sparse(ROWS[2]), rows[1][0], TOL);
    }

    @Test
    public void boundaryIdsCountOnlyAtTheEnds() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec(), weights);
        // the ids in the middle stay: only the leading and trailing rows are the pooler's filter
        assertEquals(3, head.classifyTokens(ROWS, new int[]{BOS, 7, BOS, 8}).length);
        assertEquals(4, head.classifyTokens(ROWS, new int[]{7, BOS, 8, 9}).length);
    }

    @Test
    public void theCombinedOutputIsTheEmbeddingThenTheFlattenedScores() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec(), weights);
        float[] combined = head.embedAndTokenClassify(ROWS, new int[]{BOS, 7, 8, EOS});
        float[] pooled = pooledDense(ROWS);
        assertEquals(N_IN + 2, combined.length);
        assertArrayEquals(pooled, java.util.Arrays.copyOf(combined, N_IN), TOL);
        assertEquals(sparse(ROWS[1]), combined[N_IN], TOL);
        assertEquals(sparse(ROWS[2]), combined[N_IN + 1], TOL);
    }

    @Test
    public void theGatesOfferTheEmbeddingAndBothTokenTasks() throws IOException {
        open();
        ModelSpec spec = spec();
        EncodeGate.checkEncode(spec);
        EncodeGate.checkEncodeTokens(spec);
        EncodeGate.checkTokenClassify(spec);
        assertTrue("the dense pooled embedding stays the embed output", EncodeGate.isEncoder(spec));
        assertTrue(EncodeGate.embedsPerToken(spec));
        assertTrue(EncodeGate.canTokenClassify(spec));
        assertTrue(EncodeGate.hasTokenEmbeddingProjection(spec));
        assertTrue(EncodeGate.hasTokenScoreProjection(spec));
        assertFalse("the sparse projection is not the legacy token classifier",
                EncodeGate.hasTokenClassifier(spec));
        assertFalse(EncodeGate.hasSequenceClassifier(spec));
    }

    @Test
    public void mismatchedTokenIdsRefuse() throws IOException {
        open();
        PoolingHead head = PoolingHead.of(spec(), weights);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> head.poolTokens(ROWS, new int[]{BOS, EOS}));
        assertTrue(e.getMessage(), e.getMessage().contains("token ids"));
    }

    @Test
    public void withoutTheTokenProjectionsTheTasksRefuse() throws IOException {
        open();
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.BERT)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_IN).normEps(EPS).pooling(PoolingKind.CLS).causal(false)
                .poolingNormalize(true);
        HeadSpec head = HeadSpec.builder().pooling(PoolingKind.CLS).build();
        ModelSpec spec = ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(null, N_IN).build())
                .head(head)
                .build();
        assertFalse(EncodeGate.embedsPerToken(spec));
        assertThrows(Exception.class, () -> PoolingHead.of(spec, weights).classifyTokens(ROWS, null));
    }
}
