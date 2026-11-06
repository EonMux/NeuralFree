package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
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
import static org.junit.Assert.assertTrue;

/**
 * The classifier head's activation gate ({@code use_activation}): the qwen3 reranker head softmaxes its scores
 * when the activation applies and answers the raw scores when it is skipped, and the raw scores are not
 * normalized. The qwen3 arch is the head's only built-in pooler activation, so it shows the gate.
 */
public class ClassifierHeadActivationTest {

    private static final int N_EMBD = 4;
    private static final int N_OUT = 3;
    private static final float TOL = 1e-5f;
    /** {@code cls.weight} {N_EMBD, N_OUT}: row {@code o} holds the inputs of output {@code o}. */
    private static final float[] W = {
            0.5f, -1.0f, 0.25f, 2.0f,
            1.0f, 0.0f, -0.5f, 0.75f,
            -0.25f, 1.5f, 1.0f, -1.0f};
    private static final float[] B = {0.1f, -0.2f, 0.3f};
    private static final float[] POOLED = {1.0f, 2.0f, -1.0f, 0.5f};

    private Path path;
    private GGUFFileParser parser;
    private CpuWeights weights;
    private TensorCatalog catalog;

    private void open() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(Arch.QWEN3.ggufName());
        w.tensorF32(TensorNames.weight(TensorNames.CLS), new long[]{N_EMBD, N_OUT}, W);
        w.tensorF32(TensorNames.bias(TensorNames.CLS), new long[]{N_OUT}, B);
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
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN3)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(N_EMBD).normRmsEps(1e-6f).pooling(PoolingKind.RANK).causal(true);
        HeadSpec head = HeadSpec.builder().pooling(PoolingKind.RANK)
                .classifier(catalog.get(TensorNames.weight(TensorNames.CLS)), null, null).build();
        return ModelSpec.builder(hb.build(), catalog)
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(head)
                .build();
    }

    /** {@code tanh(W x + b)} in double precision, the raw scores of the synthetic head. */
    private static float[] raw() {
        float[] y = new float[N_OUT];
        for (int o = 0; o < N_OUT; o++) {
            double acc = B[o];
            for (int i = 0; i < N_EMBD; i++) {
                acc += (double) W[o * N_EMBD + i] * POOLED[i];
            }
            y[o] = (float) Math.tanh(acc);
        }
        return y;
    }

    /** The softmax of {@code v} in double precision. */
    private static float[] softmax(float[] v) {
        double max = Double.NEGATIVE_INFINITY;
        for (float x : v) {
            max = Math.max(max, x);
        }
        double sum = 0;
        float[] p = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            p[i] = (float) Math.exp(v[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < v.length; i++) {
            p[i] = (float) (p[i] / sum);
        }
        return p;
    }

    @Test
    public void theActivationGateSoftmaxesOrSkips() throws IOException {
        open();
        ClassifierHead head = ClassifierHead.of(spec(), weights);
        assertTrue("the qwen3 head normalizes when the activation applies", head.normalized());

        assertArrayEquals(softmax(raw()), head.apply(POOLED, 0), TOL);
        assertArrayEquals(raw(), head.apply(POOLED, 0, false), TOL);
        assertArrayEquals(softmax(raw()), head.apply(POOLED, 0, true), TOL);
        double sum = 0;
        for (float x : head.apply(POOLED, 0, false)) {
            sum += x;
        }
        assertTrue("the raw scores are not probabilities: " + sum, Math.abs(sum - 1.0) > 1e-3);
    }

    @Test
    public void poolingHeadClassifyCarriesTheGate() throws IOException {
        open();
        PoolingHead pooling = PoolingHead.of(spec(), weights);
        float[][] rows = {POOLED};

        ClassifierOutput applied = pooling.classify(rows);
        assertTrue("the default applies the activation", applied.normalized());
        assertArrayEquals(softmax(raw()), applied.scores(), TOL);

        ClassifierOutput rawOut = pooling.classify(rows, false);
        assertFalse("the skipped activation leaves raw scores", rawOut.normalized());
        assertArrayEquals(raw(), rawOut.scores(), TOL);
        assertEquals("the activation keeps the argmax", applied.best(), rawOut.best());
    }
}
