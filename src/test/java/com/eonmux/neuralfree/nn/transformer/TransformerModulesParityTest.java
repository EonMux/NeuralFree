package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The nn/transformer modules on their TensorOps compositions. {@link HostPath} runs them on the CPU reference backend
 * against independent references: double-precision attention (head split and merge, causal and additive masks,
 * all-masked rows), the Java float formulas of the position encodings, the cache contents, and an independent host
 * definition of the mixture of experts (stable-sort top-k, slot-ordered accumulation). {@link Device} runs each module
 * on every available GPU API against the same module on the host backend. These are value checks of the inference
 * path, so they run with grad mode off, where no result requires grad; the recorded path computes
 * the same values, and its gradients are checked by {@link TransformerAutogradParityTest}, which also uses the
 * analytic attention gradient {@link Refs#attentionBackward}.
 *
 * <p>Tolerances follow the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. Data movement (caches,
 * masked {@code -inf} scores, exact zero weights) is bit-identical and the encodings are single-rounding ops. Attention
 * and expert outputs chain GEMMs (reduction class, K at most 16 here, terms of order 1), a softmax (transcendental
 * class) and a second GEMM; with inputs in {@code [-1, 1]} the stage bounds add up to well below {@value #CHAIN_TOL}
 * relative to {@code max(1, |ref|)}, the bound of every chained comparison.</p>
 */
@RunWith(Enclosed.class)
public class TransformerModulesParityTest {

    static final double CHAIN_TOL = 1e-4;
    static final long SEED = 4242L;
    static final float NINF = Float.NEGATIVE_INFINITY;

    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(false);
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                assertEquals("nothing is recorded with grad mode off", 0, ComputationGraph.current().size());
            } finally {
                ComputationGraph.setGradEnabled(true);
                ComputationGraph.current().clear();
                TrainGpu.release(host);
            }
        }

        @Test
        public void multiHeadAttentionMatchesReference() {
            float[] x = TrainParity.uniform(11, 2 * 4 * 8, -1f, 1f);
            float[] keyBias = {0f, -0.5f, 0.25f, 0f, NINF, 0f, 0.5f, 0f};
            for (boolean batchFirst : new boolean[]{true, false}) {
                for (boolean causal : new boolean[]{false, true}) {
                    MultiHeadAttention mha = new MultiHeadAttention(8, 2, 0.5f, causal, batchFirst);
                    mha.initialize(host, new Random(SEED));
                    mha.eval();
                    Shape layout = batchFirst ? Shape.of(2, 4, 8) : Shape.of(4, 2, 8);
                    String what = "mha batchFirst=" + batchFirst + " causal=" + causal;
                    try (Tensor in = tensor(host, x, layout.dims());
                         Tensor mask = tensor(host, keyBias, 2, 1, 1, 4);
                         Tensor out = mha.forward(in, in, in, mask)) {
                        assertEquals(what + " shape", layout, out.shape());
                        assertFalse(what + " requires grad under noGrad", out.requiresGrad());
                        assertChain(what, Refs.mha(mha, x, 2, 4, batchFirst, causal, keyBias), out.toFloatArray());
                    } finally {
                        mha.close();
                    }
                }
            }
        }

        @Test
        public void flashAttentionMasksExactlyAndMatchesReference() {
            Scenario sc = new Scenario(3, 4, false);
            try (Tensor q = sc.q(host); Tensor k = sc.k(host); Tensor v = sc.v(host); Tensor mask = sc.mask(host);
                 FlashAttentionOp.FlashAttentionResult r = FlashAttentionOp.apply(q, k, v, sc.scale, false, mask);
                 Tensor weights = TensorOps.maskedSoftmax(r.scores, 3)) {
                Refs.Attention ref = sc.reference();
                float[] scores = r.scores.toFloatArray();
                float[] p = weights.toFloatArray();
                for (int i = 0; i < scores.length; i++) {
                    if (ref.scores[i] == Double.NEGATIVE_INFINITY) {
                        assertEquals("masked score " + i, NINF, scores[i], 0f);
                        assertEquals("masked weight " + i, 0, Float.floatToRawIntBits(p[i]));
                    }
                }
                float[] out = r.output.toFloatArray();
                float[] l = r.l.toFloatArray();
                assertChain("flash output", ref.out, out);
                assertChain("flash weights", ref.p, p);
                assertChain("flash normalizer", ref.l, l);
                for (int row = sc.h * sc.sq; row < 2 * sc.h * sc.sq; row++) {
                    assertEquals("all-masked row normalizer " + row, 0f, l[row], 0f);
                    for (int c = 0; c < sc.d; c++) {
                        assertEquals("all-masked row output " + row, 0f, out[row * sc.d + c], 0f);
                    }
                }
            }
        }

        @Test
        public void kvCacheAppendsSlicesResizesAndClears() {
            float[] keys = KvScenario.run(host)[0];
            for (int i = 0; i < keys.length; i++) {
                int s = i / 3 % 3;
                float expected = s < 2 ? KvScenario.first(i / 18, s, i / 9 % 2, i % 3)
                    : KvScenario.second(i / 18, i / 9 % 2, i % 3);
                assertEquals("key " + i, expected, keys[i], 0f);
            }
        }

        @Test
        public void positionEncodingsMatchFloatFormulas() {
            float[] x = TrainParity.uniform(41, 2 * 4 * 8, -1f, 1f);
            RotaryPE rope = new RotaryPE(8, 6, 6, 10000f);
            SinusoidalPE sin = new SinusoidalPE(8, 6);
            LearnedPE learned = new LearnedPE(8, 6);
            ALiBiPE alibi = new ALiBiPE(8, 6, 3);
            float[] tokens = Arrays.copyOf(x, 5 * 8);
            try {
                rope.initialize(host);
                sin.initialize(host);
                learned.initialize(host, new Random(SEED));
                alibi.initialize(host);
                try (Tensor in = tensor(host, x, 2, 4, 8); Tensor in2 = tensor(host, tokens, 5, 8);
                     Tensor rotated = rope.apply(in, 1); Tensor sinOut = sin.apply(in, 2);
                     Tensor learnedOut = learned.apply(in2, 1); Tensor bias = alibi.getAttentionBias(3, 4, 1)) {
                    TrainParity.assertBitIdentical("rotary", Refs.rotary(rope, x, 4, 8, 6, 1), rotated.toFloatArray());
                    TrainParity.assertBitIdentical("sinusoidal",
                        Refs.added(x, sin.getPEBuffer().toFloatArray(), 4, 8, 2), sinOut.toFloatArray());
                    TrainParity.assertBitIdentical("learned",
                        Refs.added(tokens, learned.getPEEmbedding().tensor().toFloatArray(), 5, 8, 1),
                        learnedOut.toFloatArray());
                    assertEquals(Shape.of(3, 3, 4), bias.shape());
                    float[] biasValues = bias.toFloatArray();
                    TrainParity.assertBitIdentical("alibi", Refs.alibi(alibi.getSlopes(), 3, 4, 1), biasValues);
                    assertEquals(-3f * alibi.getSlope(0), biasValues[2 * 4], 0f);
                    assertEquals(-2f * alibi.getSlope(0), biasValues[2 * 4 + 1], 0f);
                    assertEquals(-1f * alibi.getSlope(0), biasValues[4 + 3], 0f);
                    IllegalArgumentException tooLong = assertThrows(IllegalArgumentException.class,
                        () -> learned.apply(in2, 2));
                    assertTrue(tooLong.getMessage().contains("exceeds maxSeqLen"));
                }
            } finally {
                rope.close();
                sin.close();
                learned.close();
                alibi.close();
            }
        }

        @Test
        public void mixtureOfExpertsMatchesHostAlgorithm() {
            MixtureOfExperts moe = new MixtureOfExperts(6, 10, 4, 2);
            moe.initialize(host, new Random(SEED));
            float[] x = TrainParity.uniform(51, 2 * 3 * 6, -1f, 1f);
            try (Tensor in = tensor(host, x, 2, 3, 6); Tensor out = moe.forward(in)) {
                assertEquals(Shape.of(2, 3, 6), out.shape());
                Refs.Moe ref = Refs.moe(moe, host, x, 6, 4, 2);
                assertArrayEquals("expert indices", ref.indices, moe.getLastExpertIndices());
                assertChain("router probs", toDouble(ref.probs), moe.getLastRouterProbs());
                assertChain("moe output", toDouble(ref.out), out.toFloatArray());
                assertEquals("load balance loss", ref.loss, moe.getLoadBalanceLoss(), 1e-5);
            } finally {
                moe.close();
            }
        }
    }

    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private ComputeBackend host;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(false);
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                assertEquals("nothing is recorded with grad mode off", 0, ComputationGraph.current().size());
            } finally {
                ComputationGraph.setGradEnabled(true);
                ComputationGraph.current().clear();
                try {
                    TrainGpu.release(host);
                } finally {
                    if (session != null) {
                        session.close();
                    }
                }
            }
        }

        @Test
        public void multiHeadAttentionMatchesCpuPath() {
            float[] x = TrainParity.uniform(61, 2 * 5 * 8, -1f, 1f);
            float[] kv = TrainParity.uniform(62, 2 * 3 * 8, -1f, 1f);
            float[] keyBias = {0f, NINF, 0.5f, -0.25f, 0f, 0f};
            for (boolean causal : new boolean[]{false, true}) {
                MultiHeadAttention ref = new MultiHeadAttention(8, 2, 0f, causal, true);
                MultiHeadAttention dev = new MultiHeadAttention(8, 2, 0f, causal, true);
                ref.initialize(host, new Random(SEED));
                dev.initialize(gpu, new Random(SEED));
                dev.loadStateDict(ref.stateDict());
                try (Tensor hx = tensor(host, x, 2, 5, 8); Tensor gx = tensor(gpu, x, 2, 5, 8);
                     Tensor hkv = tensor(host, kv, 2, 3, 8); Tensor gkv = tensor(gpu, kv, 2, 3, 8);
                     Tensor hm = tensor(host, keyBias, 2, 1, 1, 3); Tensor gm = tensor(gpu, keyBias, 2, 1, 1, 3);
                     Tensor hSelf = ref.forward(hx); Tensor gSelf = dev.forward(gx);
                     Tensor hCross = ref.forward(hx, hkv, hkv, hm); Tensor gCross = dev.forward(gx, gkv, gkv, gm)) {
                    assertSame(gpu, gSelf.backend());
                    assertChain(api + " self " + causal, toDouble(hSelf.toFloatArray()), gSelf.toFloatArray());
                    assertChain(api + " cross " + causal, toDouble(hCross.toFloatArray()), gCross.toFloatArray());
                } finally {
                    ref.close();
                    dev.close();
                }
            }
        }

        @Test
        public void flashAttentionMatchesCpuPath() {
            for (Scenario sc : new Scenario[]{new Scenario(3, 4, false), new Scenario(4, 4, true)}) {
                String what = api + (sc.causal ? " causal" : " masked");
                try (Tensor hq = sc.q(host); Tensor hk = sc.k(host); Tensor hv = sc.v(host);
                     Tensor gq = sc.q(gpu); Tensor gk = sc.k(gpu); Tensor gv = sc.v(gpu);
                     Tensor hm = sc.mask(host); Tensor gm = sc.mask(gpu);
                     FlashAttentionOp.FlashAttentionResult hr =
                         FlashAttentionOp.apply(hq, hk, hv, sc.scale, sc.causal, hm);
                     FlashAttentionOp.FlashAttentionResult gr =
                         FlashAttentionOp.apply(gq, gk, gv, sc.scale, sc.causal, gm)) {
                    assertSame(gpu, gr.output.backend());
                    assertChain(what + " scores", toDouble(hr.scores.toFloatArray()), gr.scores.toFloatArray());
                    assertChain(what + " output", toDouble(hr.output.toFloatArray()), gr.output.toFloatArray());
                    assertChain(what + " normalizer", toDouble(hr.l.toFloatArray()), gr.l.toFloatArray());
                }
            }
        }

        @Test
        public void kvCacheIsBitIdentical() {
            float[][] ref = KvScenario.run(host);
            float[][] dev = KvScenario.run(gpu);
            for (int i = 0; i < ref.length; i++) {
                TrainParity.assertBitIdentical(api + " kv cache snapshot " + i, ref[i], dev[i]);
            }
        }

        @Test
        public void positionEncodingsMatchCpuPath() {
            float[] x = TrainParity.uniform(81, 2 * 4 * 8, -1f, 1f);
            ComputeBackend[] backends = {host, gpu};
            float[][][] results = new float[2][][];
            for (int i = 0; i < 2; i++) {
                RotaryPE rope = new RotaryPE(8, 6, 6, 10000f);
                SinusoidalPE sin = new SinusoidalPE(8, 6);
                LearnedPE learned = new LearnedPE(8, 6);
                ALiBiPE alibi = new ALiBiPE(8, 6, 3);
                try {
                    rope.initialize(backends[i]);
                    sin.initialize(backends[i]);
                    learned.initialize(backends[i], new Random(SEED));
                    alibi.initialize(backends[i]);
                    try (Tensor in = tensor(backends[i], x, 2, 4, 8);
                         Tensor r = rope.apply(in, 2); Tensor s = sin.apply(in, 1); Tensor l = learned.apply(in, 2);
                         Tensor b = alibi.getAttentionBias(4, 5, 2)) {
                        assertSame(backends[i], r.backend());
                        results[i] = new float[][]{r.toFloatArray(), s.toFloatArray(), l.toFloatArray(),
                            b.toFloatArray()};
                    }
                } finally {
                    rope.close();
                    sin.close();
                    learned.close();
                    alibi.close();
                }
            }
            String[] names = {"rotary", "sinusoidal", "learned", "alibi"};
            for (int k = 0; k < names.length; k++) {
                TrainParity.assertSingleRounding(api + " " + names[k], results[0][k], results[1][k]);
            }
        }

        @Test
        public void mixtureOfExpertsMatchesCpuPath() {
            MixtureOfExperts ref = new MixtureOfExperts(6, 10, 4, 2);
            MixtureOfExperts dev = new MixtureOfExperts(6, 10, 4, 2);
            ref.initialize(host, new Random(SEED));
            dev.initialize(gpu, new Random(SEED));
            dev.loadStateDict(ref.stateDict());
            float[] x = TrainParity.uniform(91, 3 * 4 * 6, -1f, 1f);
            try (Tensor hx = tensor(host, x, 3, 4, 6); Tensor gx = tensor(gpu, x, 3, 4, 6);
                 Tensor ho = ref.forward(hx); Tensor go = dev.forward(gx)) {
                assertSame(gpu, go.backend());
                assertArrayEquals(api + " expert indices", ref.getLastExpertIndices(), dev.getLastExpertIndices());
                TrainParity.assertTranscendental(api + " router probs", ref.getLastRouterProbs(),
                    dev.getLastRouterProbs());
                assertChain(api + " moe output", toDouble(ho.toFloatArray()), go.toFloatArray());
                assertEquals(api + " load balance loss", ref.getLoadBalanceLoss(), dev.getLoadBalanceLoss(), 1e-5);
            } finally {
                ref.close();
                dev.close();
            }
        }
    }

    // ------------------------------------------------------------------ scenarios

    /** A fixed {@code [b, h, s, d]} attention: causal and unmasked, or with a key mask fully masking batch 1. */
    static final class Scenario {
        final int b = 2, h = 2, d = 3;
        final int sq, skv;
        final boolean causal;
        final float scale = 1.0f / (float) Math.sqrt(3);
        final float[] q, k, v, keyBias;

        Scenario(int sq, int skv, boolean causal) {
            this.sq = sq;
            this.skv = skv;
            this.causal = causal;
            this.q = TrainParity.uniform(21, b * h * sq * d, -1f, 1f);
            this.k = TrainParity.uniform(22, b * h * skv * d, -1f, 1f);
            this.v = TrainParity.uniform(23, b * h * skv * d, -1f, 1f);
            this.keyBias = causal ? null : new float[]{0f, NINF, 0.5f, -0.25f, NINF, NINF, NINF, NINF};
        }

        Tensor q(ComputeBackend be) {
            return tensor(be, q, b, h, sq, d);
        }

        Tensor k(ComputeBackend be) {
            return tensor(be, k, b, h, skv, d);
        }

        Tensor v(ComputeBackend be) {
            return tensor(be, v, b, h, skv, d);
        }

        /** The additive key mask {@code [b, 1, 1, skv]}, or null for the causal scenario. */
        Tensor mask(ComputeBackend be) {
            return keyBias == null ? null : tensor(be, keyBias, b, 1, 1, skv);
        }

        Refs.Attention reference() {
            return Refs.attention(toDouble(q), toDouble(k), toDouble(v), b, h, sq, skv, d, scale, causal, keyBias);
        }
    }

    /** One cache history: two positions through a permuted view, one through a contiguous tensor, resize, clear. */
    static final class KvScenario {
        static float first(int b, int s, int h, int d) {
            return 100 * b + 10 * s + h + 0.125f * d;
        }

        static float second(int b, int h, int d) {
            return -(100 * b + h + 0.25f * d);
        }

        /** {@code [keys after 3 positions, values after 3, keys after resize(2), first keys read after clear]}. */
        static float[][] run(ComputeBackend be) {
            float[] firstData = new float[2 * 2 * 2 * 3];
            float[] secondData = new float[2 * 2 * 3];
            for (int i = 0; i < firstData.length; i++) {
                firstData[i] = first(i / 12, i / 6 % 2, i / 3 % 2, i % 3);
                if (i < secondData.length) {
                    secondData[i] = second(i / 6, i / 3 % 2, i % 3);
                }
            }
            KVCache cache = new KVCache(2, 2, 3, 4, 2);
            cache.initialize(be);
            try (Tensor bshd = tensor(be, firstData, 2, 2, 2, 3);
                 Tensor firstView = bshd.permute(0, 2, 1, 3);
                 Tensor secondT = tensor(be, secondData, 2, 2, 1, 3)) {
                cache.append(0, firstView, firstView);
                cache.append(0, secondT, secondT);
                assertEquals(3, cache.getSeqLength(0));
                assertNull(cache.getKeys(1));
                assertThrows(IllegalStateException.class, () -> cache.append(0, firstView, firstView));
                float[][] out = new float[4][];
                try (Tensor keys = cache.getKeys(0); Tensor values = cache.getValues(0)) {
                    assertEquals(Shape.of(2, 2, 3, 3), keys.shape());
                    out[0] = keys.toFloatArray();
                    out[1] = values.toFloatArray();
                    cache.resize(2);
                    assertEquals(2, cache.maxSeqLen());
                    assertEquals(2, cache.getSeqLength(0));
                    try (Tensor truncated = cache.getKeys(0)) {
                        out[2] = truncated.toFloatArray();
                    }
                    cache.clear();
                    assertNull(cache.getKeys(0));
                    out[3] = keys.toFloatArray();
                    TrainParity.assertBitIdentical("snapshot survives clear", out[0], out[3]);
                }
                return out;
            } finally {
                cache.close();
            }
        }
    }

    // ------------------------------------------------------------------ references

    /** Double-precision attention and Java float definitions of the encodings and the expert routing. */
    static final class Refs {

        static final class Attention {
            double[] scores, p, out, l;
        }

        static Attention attention(double[] q, double[] k, double[] v, int b, int h, int sq, int skv, int d,
                                   float scale, boolean causal, float[] keyBias) {
            Attention a = new Attention();
            a.scores = new double[b * h * sq * skv];
            a.p = new double[a.scores.length];
            a.out = new double[b * h * sq * d];
            a.l = new double[b * h * sq];
            for (int row = 0; row < b * h * sq; row++) {
                int bh = row / sq;
                int i = row % sq;
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < skv; j++) {
                    double bias = keyBias == null ? 0.0 : keyBias[bh / h * skv + j];
                    double dot = 0.0;
                    for (int c = 0; c < d; c++) {
                        dot += q[row * d + c] * k[(bh * skv + j) * d + c];
                    }
                    double score = (causal && j > i) || bias == Double.NEGATIVE_INFINITY
                        ? Double.NEGATIVE_INFINITY : dot * scale + bias;
                    a.scores[row * skv + j] = score;
                    max = Math.max(max, score);
                }
                for (int j = 0; j < skv; j++) {
                    double s = a.scores[row * skv + j];
                    a.p[row * skv + j] = s == Double.NEGATIVE_INFINITY ? 0.0 : Math.exp(s - max);
                    a.l[row] += a.p[row * skv + j];
                }
                for (int j = 0; j < skv; j++) {
                    a.p[row * skv + j] = a.l[row] == 0.0 ? 0.0 : a.p[row * skv + j] / a.l[row];
                    for (int c = 0; c < d; c++) {
                        a.out[row * d + c] += a.p[row * skv + j] * v[(bh * skv + j) * d + c];
                    }
                }
            }
            return a;
        }

        /** {@code [dQ, dK, dV]} of {@link #attention}'s output for the upstream gradient {@code gOut}. */
        static double[][] attentionBackward(Attention a, Scenario sc, float[] gOut) {
            int skv = sc.skv;
            int d = sc.d;
            double[][] g = {new double[sc.q.length], new double[sc.k.length], new double[sc.v.length]};
            for (int row = 0; row < sc.b * sc.h * sc.sq; row++) {
                int bh = row / sc.sq;
                double[] dp = new double[skv];
                double dot = 0.0;
                for (int j = 0; j < skv; j++) {
                    for (int c = 0; c < d; c++) {
                        dp[j] += gOut[row * d + c] * (double) sc.v[(bh * skv + j) * d + c];
                        g[2][(bh * skv + j) * d + c] += a.p[row * skv + j] * gOut[row * d + c];
                    }
                    dot += a.p[row * skv + j] * dp[j];
                }
                for (int j = 0; j < skv; j++) {
                    double ds = a.p[row * skv + j] * (dp[j] - dot) * sc.scale;
                    for (int c = 0; c < d; c++) {
                        g[0][row * d + c] += ds * sc.k[(bh * skv + j) * d + c];
                        g[1][(bh * skv + j) * d + c] += ds * sc.q[row * d + c];
                    }
                }
            }
            return g;
        }

        /** Self-attention of {@code mha} over token rows {@code x} in its layout, from its projection weights. */
        static double[] mha(MultiHeadAttention mha, float[] x, int batch, int seq, boolean batchFirst,
                            boolean causal, float[] keyBias) {
            int dm = mha.dModel();
            int h = mha.numHeads();
            int d = dm / h;
            Linear[] proj = {mha.qProj(), mha.kProj(), mha.vProj(), mha.outProj()};
            double[][] heads = new double[3][batch * h * seq * d];
            for (int t = 0; t < batch * seq; t++) {
                int b = t / seq;
                int s = t % seq;
                int row = batchFirst ? t : s * batch + b;
                for (int p = 0; p < 3; p++) {
                    double[] y = linear(proj[p], toDouble(x), row, dm);
                    for (int o = 0; o < dm; o++) {
                        heads[p][((b * h + o / d) * seq + s) * d + o % d] = y[o];
                    }
                }
            }
            Attention a = attention(heads[0], heads[1], heads[2], batch, h, seq, seq, d,
                1.0f / (float) Math.sqrt(d), causal, keyBias);
            double[] merged = new double[batch * seq * dm];
            for (int t = 0; t < batch * seq; t++) {
                int row = batchFirst ? t : (t % seq) * batch + t / seq;
                for (int o = 0; o < dm; o++) {
                    merged[row * dm + o] = a.out[((t / seq * h + o / d) * seq + t % seq) * d + o % d];
                }
            }
            double[] out = new double[merged.length];
            for (int row = 0; row < batch * seq; row++) {
                System.arraycopy(linear(proj[3], merged, row, dm), 0, out, row * dm, dm);
            }
            return out;
        }

        private static double[] linear(Linear layer, double[] x, int row, int dm) {
            float[] w = layer.weight().tensor().toFloatArray();
            float[] bias = layer.bias().tensor().toFloatArray();
            double[] y = new double[dm];
            for (int o = 0; o < dm; o++) {
                y[o] = bias[o];
                for (int i = 0; i < dm; i++) {
                    y[o] += x[row * dm + i] * w[o * dm + i];
                }
            }
            return y;
        }

        /** RotaryPE's formula in Java float arithmetic over its own caches. */
        static float[] rotary(RotaryPE rope, float[] x, int seq, int dm, int rotaryDim, int offset) {
            float[] cos = rope.getCosCache().toFloatArray();
            float[] sin = rope.getSinCache().toFloatArray();
            int half = rotaryDim / 2;
            float[] out = x.clone();
            for (int t = 0; t < x.length / dm; t++) {
                int pos = offset + t % seq;
                for (int i = 0; i < half; i++) {
                    float v1 = x[t * dm + i];
                    float v2 = x[t * dm + i + half];
                    float c = cos[pos * half + i];
                    float s = sin[pos * half + i];
                    out[t * dm + i] = v1 * c - v2 * s;
                    out[t * dm + i + half] = v1 * s + v2 * c;
                }
            }
            return out;
        }

        /** {@code x + table[offset + s]} per token row. */
        static float[] added(float[] x, float[] table, int seq, int dm, int offset) {
            float[] out = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                out[i] = x[i] + table[(offset + i / dm % seq) * dm + i % dm];
            }
            return out;
        }

        static float[] alibi(float[] slopes, int queryLen, int keyLen, int offset) {
            float[] out = new float[slopes.length * queryLen * keyLen];
            for (int i = 0; i < out.length; i++) {
                int distance = Math.abs(i % keyLen - (offset + i / keyLen % queryLen));
                out[i] = -slopes[i / (keyLen * queryLen)] * distance;
            }
            return out;
        }

        static final class Moe {
            int[] indices;
            float[] probs, out;
            float loss;
        }

        /**
         * A host definition of MixtureOfExperts over the module's router and experts: per token, a stable descending
         * sort of the router probabilities, gates normalized by their sum, each expert run on the single token,
         * outputs accumulated in slot order from 0; the load-balance loss in sequential float order.
         */
        static Moe moe(MixtureOfExperts m, ComputeBackend be, float[] x, int dm, int experts, int k) {
            int tokens = x.length / dm;
            Moe r = new Moe();
            try (Tensor in = tensor(be, x, tokens, dm); Tensor logits = m.router().forward(in);
                 Tensor probs = TensorOps.softmax(logits, 1)) {
                r.probs = probs.toFloatArray();
            }
            r.indices = new int[tokens * k];
            r.out = new float[tokens * dm];
            float[] counts = new float[experts];
            float[] probSums = new float[experts];
            for (int t = 0; t < tokens; t++) {
                final int base = t * experts;
                Integer[] order = new Integer[experts];
                float total = 0.0f;
                for (int e = 0; e < experts; e++) {
                    order[e] = e;
                    probSums[e] += r.probs[base + e];
                }
                Arrays.sort(order, (a, b) -> Float.compare(r.probs[base + b], r.probs[base + a]));
                for (int s = 0; s < k; s++) {
                    total += r.probs[base + order[s]];
                }
                for (int s = 0; s < k; s++) {
                    int e = order[s];
                    r.indices[t * k + s] = e;
                    counts[e] += 1.0f;
                    float gate = total > 0 ? r.probs[base + e] / total : r.probs[base + e];
                    try (Tensor in = Tensor.fromArray(be, Arrays.copyOfRange(x, t * dm, (t + 1) * dm));
                         Tensor y = m.experts().get(e).forward(in)) {
                        float[] yv = y.toFloatArray();
                        for (int c = 0; c < dm; c++) {
                            r.out[t * dm + c] += gate * yv[c];
                        }
                    }
                }
            }
            for (int e = 0; e < experts; e++) {
                r.loss += experts * (probSums[e] / tokens) * (counts[e] / (tokens * k));
            }
            return r;
        }
    }

    // ------------------------------------------------------------------ helpers

    static Tensor tensor(ComputeBackend be, float[] data, int... dims) {
        try (Tensor flat = Tensor.fromArray(be, data)) {
            return flat.reshape(Shape.of(dims));
        }
    }

    static double[] toDouble(float[] f) {
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    /** {@code |got - ref| <= CHAIN_TOL * max(1, |ref|)}; infinities and NaN must match exactly. */
    static void assertChain(String what, double[] ref, float[] got) {
        assertEquals(what + ": length", ref.length, got.length);
        for (int i = 0; i < ref.length; i++) {
            boolean finite = Double.isFinite(ref[i]) && Float.isFinite(got[i]);
            boolean ok = finite ? Math.abs(got[i] - ref[i]) <= CHAIN_TOL * Math.max(1.0, Math.abs(ref[i]))
                : Double.compare(ref[i], got[i]) == 0;
            if (!ok) {
                fail(what + ": element " + i + ": ref " + ref[i] + ", got " + got[i]);
            }
        }
    }
}
