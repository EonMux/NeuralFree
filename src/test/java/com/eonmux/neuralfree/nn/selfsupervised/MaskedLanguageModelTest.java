package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Embedding;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link MaskedLanguageModel}'s BERT prediction head ({@code dense -> GELU -> LayerNorm(eps 1e-12) -> decoder}), on the
 * CPU path and on each GPU API: the registered children and parameters, untied and tied to the encoder's embedding
 * table; logits, loss and the decoder gradients against an independent double-precision host computation from the
 * module's own parameter values ({@code |got - ref| <= 1e-4 * max(1, |ref|)}, a composition class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}); gradients that reach the head and the encoder; and
 * a tied weight whose gradient is the sum of its two uses.
 */
@RunWith(Enclosed.class)
public class MaskedLanguageModelTest {

    static final double TOL = 1e-4;
    static final int VOCAB = 10;
    static final int HIDDEN = 6;
    static final int BATCH = 2;
    static final int SEQ = 5;
    static final int[] IDS = {0, 3, 5, 2, 9, 0, 1, 8, 4, 9};
    static final int[] POSITIONS = {1, 3, 6, 8};
    static final int[] TARGETS = {2, 7, 1, 9};

    /** The CPU path, on a backend without the train library. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            TrainGpu.release(host);
        }

        @Test
        public void headIsRegisteredUntiedAndTied() {
            Checks.registration(host);
        }

        @Test
        public void invalidTiesAndUseBeforeInitializationAreRejected() {
            Checks.rejections(host);
        }

        @Test
        public void lossMatchesHostReference() {
            Checks.reference(host, "cpu path", false);
            Checks.reference(host, "cpu path", true);
        }

        @Test
        public void gradientReachesHeadAndEncoder() {
            Checks.gradientReach(host, "cpu path");
        }

        @Test
        public void tiedGradientSumsBothUses() {
            Checks.tiedGradient(host, "cpu path");
        }
    }

    /** The device path on each GPU API, against the same host computation. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            gpu = session.backend();
        }

        @After
        public void tearDown() {
            if (session != null) {
                session.close();
            }
        }

        @Test
        public void lossMatchesHostReference() {
            Checks.reference(gpu, api.toString(), false);
            Checks.reference(gpu, api.toString(), true);
        }

        @Test
        public void gradientReachesHeadAndEncoder() {
            Checks.gradientReach(gpu, api.toString());
        }

        @Test
        public void tiedGradientSumsBothUses() {
            Checks.tiedGradient(gpu, api.toString());
        }
    }

    static final class Checks {
        private Checks() {
        }

        /** An initialized model over a seeded embedding encoder, tied or not. */
        static MaskedLanguageModel model(ComputeBackend b, boolean tied) {
            Embedding embedding = new Embedding(VOCAB, HIDDEN);
            embedding.initialize(b, new Random(61));
            MaskedLanguageModel mlm = tied ? new MaskedLanguageModel(embedding, VOCAB, HIDDEN, embedding.weight())
                : new MaskedLanguageModel(embedding, VOCAB, HIDDEN);
            mlm.setRandom(new Random(62));
            mlm.initialize(b);
            return mlm;
        }

        static MaskedLanguageModel.MLMMask mask() {
            return new MaskedLanguageModel.MLMMask(POSITIONS, TARGETS, SEQ, 0.15f);
        }

        static Tensor ids(ComputeBackend b) {
            try (Tensor flat = Tensor.fromArray(b, IDS)) {
                return flat.reshape(BATCH, SEQ);
            }
        }

        static void registration(ComputeBackend b) {
            try (MaskedLanguageModel mlm = model(b, false)) {
                assertEquals("children", Arrays.asList("encoder", "dense", "activation", "layer_norm", "decoder"),
                    new ArrayList<>(mlm.namedChildren().keySet()));
                Map<String, Parameter> params = mlm.namedParameters();
                assertEquals("untied parameters", Arrays.asList("encoder.weight", "dense.weight", "dense.bias",
                    "layer_norm.weight", "layer_norm.bias", "decoder.weight", "decoder.bias"),
                    new ArrayList<>(params.keySet()));
                assertEquals(Shape.of(HIDDEN, HIDDEN), params.get("dense.weight").shape());
                assertEquals(Shape.vector(HIDDEN), params.get("dense.bias").shape());
                assertEquals(Shape.vector(HIDDEN), params.get("layer_norm.weight").shape());
                assertEquals(Shape.of(VOCAB, HIDDEN), params.get("decoder.weight").shape());
                assertEquals(Shape.vector(VOCAB), params.get("decoder.bias").shape());
                assertEquals("BERT's layer-norm epsilon", 1e-12f, mlm.layerNorm().epsilon(), 0.0f);
                assertTrue("the untied decoder is a Linear", mlm.decoder() instanceof Linear);
                assertFalse(mlm.isDecoderTied());
                assertArrayEquals("zero decoder bias", new float[VOCAB], mlm.decoderBias().tensor().toFloatArray(),
                    0.0f);
            }
            try (MaskedLanguageModel mlm = model(b, true)) {
                Parameter table = ((Embedding) mlm.encoder()).weight();
                assertTrue(mlm.isDecoderTied());
                assertSame("the decoder reads the embedding table", table, mlm.decoderWeight());
                assertEquals("tied parameters", Arrays.asList("encoder.weight", "dense.weight", "dense.bias",
                    "layer_norm.weight", "layer_norm.bias", "decoder.bias"),
                    new ArrayList<>(mlm.namedParameters().keySet()));
                assertEquals("the tied table is listed once", 1,
                    Collections.frequency(mlm.parameters(), table));
                assertEquals("parameter count", VOCAB * HIDDEN + HIDDEN * HIDDEN + 3 * HIDDEN + VOCAB,
                    mlm.parameterCount());
            }
        }

        static void rejections(ComputeBackend b) {
            Embedding embedding = new Embedding(VOCAB, HIDDEN);
            embedding.initialize(b, new Random(63));
            Embedding stranger = new Embedding(VOCAB, HIDDEN);
            stranger.initialize(b, new Random(64));
            Embedding narrow = new Embedding(VOCAB, HIDDEN - 1);
            narrow.initialize(b, new Random(65));
            try {
                expectIllegalArgument("a weight outside the encoder",
                    () -> new MaskedLanguageModel(embedding, VOCAB, HIDDEN, stranger.weight()));
                expectIllegalArgument("a weight of the wrong shape",
                    () -> new MaskedLanguageModel(narrow, VOCAB, HIDDEN, narrow.weight()));
            } finally {
                stranger.close();
                narrow.close();
            }
            try (MaskedLanguageModel mlm = new MaskedLanguageModel(embedding, VOCAB, HIDDEN);
                 Tensor ids = ids(b)) {
                try {
                    mlm.forwardMLM(ids, mask());
                    fail("forwardMLM before initialize: expected IllegalStateException");
                } catch (IllegalStateException expected) {
                    assertTrue(expected.getMessage(), expected.getMessage().contains("initialize"));
                }
                mlm.initialize(b);
                try {
                    mlm.initialize(b);
                    fail("a second initialize: expected IllegalStateException");
                } catch (IllegalStateException expected) {
                    assertTrue(expected.getMessage(), expected.getMessage().contains("already initialized"));
                }
            }
        }

        /**
         * One forward under the fixed seeds: logits and loss against {@link HostReference}, predictions equal to the
         * first maximum of each returned logit row, and the decoder gradients of the loss against the reference.
         */
        static void reference(ComputeBackend b, String what, boolean tied) {
            String label = what + (tied ? " tied" : " untied");
            int before = ComputationGraph.current().size();
            try (MaskedLanguageModel mlm = model(b, tied);
                 Tensor ids = ids(b)) {
                HostReference ref = new HostReference(mlm);
                MaskedLanguageModel.MLMOutput out = mlm.forwardMLM(ids, mask());
                float[] logits;
                try (Tensor l = out.logits(); Tensor predictions = out.predictions(); Tensor loss = out.mlmLoss()) {
                    assertTrue(label + ": logits are recorded", l.requiresGrad());
                    assertFalse(label + ": predictions are not recorded", predictions.requiresGrad());
                    logits = l.toFloatArray();
                    assertClose(label + " logits", ref.logits, logits);
                    assertClose(label + " loss", new double[]{ref.loss}, loss.toFloatArray());
                    assertArrayEquals(label + " predictions", firstMaxima(logits), predictions.toFloatArray(), 0.0f);
                    loss.backward();
                }
                assertEquals(label + ": graph size after backward", before, ComputationGraph.current().size());
                assertClose(label + " decoder bias gradient", ref.biasGradient,
                    mlm.decoderBias().grad().toFloatArray());
                if (!tied) {
                    assertClose(label + " decoder weight gradient", ref.weightGradient,
                        mlm.decoderWeight().grad().toFloatArray());
                }
            }
        }

        /** Every parameter of the head and the encoder table receive a finite, non-zero gradient. */
        static void gradientReach(ComputeBackend b, String what) {
            try (MaskedLanguageModel mlm = model(b, false);
                 Tensor ids = ids(b)) {
                MaskedLanguageModel.MLMOutput out = mlm.forwardMLM(ids, mask());
                try (Tensor l = out.logits(); Tensor predictions = out.predictions(); Tensor loss = out.mlmLoss()) {
                    loss.backward();
                }
                for (Map.Entry<String, Parameter> e : mlm.namedParameters().entrySet()) {
                    Tensor grad = e.getValue().grad();
                    assertNotNull(what + " " + e.getKey() + " gradient", grad);
                    assertEquals(what + " " + e.getKey() + " gradient shape", e.getValue().shape(), grad.shape());
                    boolean nonZero = false;
                    for (float v : grad.toFloatArray()) {
                        assertTrue(what + " " + e.getKey() + " gradient " + v, Float.isFinite(v));
                        nonZero |= v != 0.0f;
                    }
                    assertTrue(what + " " + e.getKey() + " gradient is not all zero", nonZero);
                }
            }
        }

        /**
         * A tied model and an untied one whose decoder weight holds the embedding table's values (same seeds, so the
         * same dense layer): equal logits, and the tied table's gradient equals the untied table's gradient plus the
         * untied decoder weight's gradient.
         */
        static void tiedGradient(ComputeBackend b, String what) {
            try (MaskedLanguageModel untied = model(b, false);
                 MaskedLanguageModel tied = model(b, true);
                 Tensor ids = ids(b)) {
                float[] table = tied.decoderWeight().tensor().toFloatArray();
                untied.loadStateDict(Collections.<String, Object>singletonMap("decoder.weight", table));
                float[] untiedLogits = backward(untied, ids);
                float[] tiedLogits = backward(tied, ids);
                assertClose(what + " tied logits vs untied", toDouble(untiedLogits), tiedLogits);
                float[] encoderPart = untied.getParameter("encoder.weight").grad().toFloatArray();
                float[] decoderPart = untied.decoderWeight().grad().toFloatArray();
                double[] sum = new double[encoderPart.length];
                for (int i = 0; i < sum.length; i++) {
                    sum[i] = (double) encoderPart[i] + decoderPart[i];
                }
                assertClose(what + " tied table gradient", sum, tied.decoderWeight().grad().toFloatArray());
            }
        }

        /** Back-propagates the model's loss and returns its logits. */
        private static float[] backward(MaskedLanguageModel mlm, Tensor ids) {
            MaskedLanguageModel.MLMOutput out = mlm.forwardMLM(ids, mask());
            try (Tensor l = out.logits(); Tensor predictions = out.predictions(); Tensor loss = out.mlmLoss()) {
                float[] logits = l.toFloatArray();
                loss.backward();
                return logits;
            }
        }

        private static void expectIllegalArgument(String what, Runnable construct) {
            try {
                construct.run();
                fail(what + ": expected IllegalArgumentException");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("tied decoder weight"));
            }
        }

        /** Per row of {@code [numMasked, VOCAB]} logits, the index of its first maximum. */
        private static float[] firstMaxima(float[] logits) {
            float[] best = new float[logits.length / VOCAB];
            for (int i = 0; i < best.length; i++) {
                int arg = 0;
                for (int v = 1; v < VOCAB; v++) {
                    if (logits[i * VOCAB + v] > logits[i * VOCAB + arg]) {
                        arg = v;
                    }
                }
                best[i] = arg;
            }
            return best;
        }

        private static double[] toDouble(float[] x) {
            double[] r = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                r[i] = x[i];
            }
            return r;
        }

        static void assertClose(String what, double[] ref, float[] got) {
            assertEquals(what + ": length", ref.length, got.length);
            for (int i = 0; i < ref.length; i++) {
                double bound = TOL * Math.max(1.0, Math.abs(ref[i]));
                if (!(Math.abs(got[i] - ref[i]) <= bound)) {
                    fail(what + ": element " + i + ": got " + got[i] + ", reference " + ref[i]);
                }
            }
        }
    }

    /**
     * The model's forward and the decoder gradients in double precision on the host, from the parameter values the
     * model holds: rows {@code h} of the embedding table at the masked positions, {@code y = h W1^T + b1}, the tanh
     * GELU {@code 0.5 y (1 + tanh(sqrt(2/pi) (y + 0.044715 y^3)))}, the layer norm {@code (g - mean) / sqrt(var +
     * 1e-12) * gamma + beta} with the biased variance, {@code logits = z Wd^T + bd}, the loss
     * {@code -mean_i(logits[i, t_i] - logsumexp(logits[i]))}, {@code dL/dbd = mean_i(softmax_i - onehot(t_i))} and
     * {@code dL/dWd = mean_i((softmax_i - onehot(t_i)) z_i^T)}.
     */
    static final class HostReference {
        final double[] logits;
        final double loss;
        final double[] biasGradient;
        final double[] weightGradient;

        HostReference(MaskedLanguageModel mlm) {
            float[] table = values(mlm, "encoder.weight");
            float[] w1 = values(mlm, "dense.weight");
            float[] b1 = values(mlm, "dense.bias");
            float[] gamma = values(mlm, "layer_norm.weight");
            float[] beta = values(mlm, "layer_norm.bias");
            float[] wd = mlm.decoderWeight().tensor().toFloatArray();
            float[] bd = mlm.decoderBias().tensor().toFloatArray();
            int m = POSITIONS.length;
            logits = new double[m * VOCAB];
            biasGradient = new double[VOCAB];
            weightGradient = new double[VOCAB * HIDDEN];
            double total = 0;
            for (int i = 0; i < m; i++) {
                int token = IDS[POSITIONS[i]];
                double[] g = new double[HIDDEN];
                for (int o = 0; o < HIDDEN; o++) {
                    double y = b1[o];
                    for (int k = 0; k < HIDDEN; k++) {
                        y += (double) table[token * HIDDEN + k] * w1[o * HIDDEN + k];
                    }
                    g[o] = 0.5 * y * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (y + 0.044715 * y * y * y)));
                }
                double[] z = layerNorm(g, gamma, beta);
                double max = Double.NEGATIVE_INFINITY;
                for (int v = 0; v < VOCAB; v++) {
                    double l = bd[v];
                    for (int k = 0; k < HIDDEN; k++) {
                        l += z[k] * wd[v * HIDDEN + k];
                    }
                    logits[i * VOCAB + v] = l;
                    max = Math.max(max, l);
                }
                double sum = 0;
                for (int v = 0; v < VOCAB; v++) {
                    sum += Math.exp(logits[i * VOCAB + v] - max);
                }
                double logSumExp = max + Math.log(sum);
                total += logits[i * VOCAB + TARGETS[i]] - logSumExp;
                for (int v = 0; v < VOCAB; v++) {
                    double d = (Math.exp(logits[i * VOCAB + v] - logSumExp) - (v == TARGETS[i] ? 1.0 : 0.0)) / m;
                    biasGradient[v] += d;
                    for (int k = 0; k < HIDDEN; k++) {
                        weightGradient[v * HIDDEN + k] += d * z[k];
                    }
                }
            }
            loss = -total / m;
        }

        private static double[] layerNorm(double[] g, float[] gamma, float[] beta) {
            double mean = 0;
            for (double v : g) {
                mean += v;
            }
            mean /= g.length;
            double var = 0;
            for (double v : g) {
                var += (v - mean) * (v - mean);
            }
            var /= g.length;
            double inv = 1.0 / Math.sqrt(var + MaskedLanguageModel.LAYER_NORM_EPSILON);
            double[] z = new double[g.length];
            for (int k = 0; k < g.length; k++) {
                z[k] = (g[k] - mean) * inv * gamma[k] + beta[k];
            }
            return z;
        }

        private static float[] values(MaskedLanguageModel mlm, String name) {
            Parameter p = mlm.getParameter(name);
            assertNotNull(name, p);
            return p.tensor().toFloatArray();
        }
    }
}
