package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.transformer.TransformerModulesParityTest.Scenario;
import com.eonmux.neuralfree.optim.SGD;
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
import java.util.Map;
import java.util.Random;

import static com.eonmux.neuralfree.nn.transformer.TransformerModulesParityTest.assertChain;
import static com.eonmux.neuralfree.nn.transformer.TransformerModulesParityTest.tensor;
import static com.eonmux.neuralfree.nn.transformer.TransformerModulesParityTest.toDouble;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The nn/transformer modules in grad mode. {@link HostPath} checks on the CPU reference backend
 * that each forward records and that one backward pass releases what it recorded (the graph size returns to its
 * pre-step value): the attention gradients against the analytic double-precision reference, every projection and
 * block parameter receiving a gradient, the learned position table's row gradients, the detached KV cache, and the
 * mixture of experts' routed gradients and auxiliary load-balance term. {@link Device} compares the same gradients on
 * every available GPU API with the host path under the chained tolerance of {@link TransformerModulesParityTest}.
 */
@RunWith(Enclosed.class)
public class TransformerAutogradParityTest {

    static final long SEED = 7171L;

    public static class HostPath {
        private ComputeBackend host;
        private ComputationGraph graph;

        @Before
        public void setUp() {
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(true);
            host = TrainParity.hostBackend();
            graph = ComputationGraph.current();
        }

        @After
        public void tearDown() {
            ComputationGraph.setGradEnabled(true);
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }

        @Test
        public void flashAttentionGradientsMatchReference() {
            Scenario[] scenarios = {new Scenario(3, 4, false), new Scenario(4, 4, true)};
            for (Scenario sc : scenarios) {
                String what = sc.causal ? "causal" : "masked";
                float[] gOut = TrainParity.uniform(31, sc.q.length, -1f, 1f);
                Tensor[] leaves = attentionLeaves(host, sc);
                int before = graph.size();
                try (Tensor mask = sc.mask(host); Tensor g = tensor(host, gOut, sc.b, sc.h, sc.sq, sc.d);
                     FlashAttentionOp.FlashAttentionResult r =
                         FlashAttentionOp.apply(leaves[0], leaves[1], leaves[2], sc.scale, sc.causal, mask)) {
                    assertTrue(what + " output is recorded", r.output.requiresGrad());
                    assertNotNull(what + " output has a gradient function", r.output.gradFn());
                    assertFalse(what + " scores are detached", r.scores.requiresGrad());
                    assertFalse(what + " normalizer is detached", r.l.requiresGrad());
                    float[] scores = r.scores.toFloatArray();

                    r.output.backward(g);

                    assertEquals(what + " backward releases the attention graph", before, graph.size());
                    assertArrayEquals(what + " scores stay valid after backward", scores, r.scores.toFloatArray(), 0f);
                    double[][] ref = TransformerModulesParityTest.Refs.attentionBackward(sc.reference(), sc, gOut);
                    for (int i = 0; i < 3; i++) {
                        assertNotNull(what + " grad " + i, leaves[i].grad());
                        assertEquals(what + " grad " + i + " shape", leaves[i].shape(), leaves[i].grad().shape());
                        assertChain(what + " grad " + i, ref[i], leaves[i].grad().toFloatArray());
                    }
                } finally {
                    closeLeaves(leaves);
                }
            }
        }

        @Test
        public void multiHeadAttentionGradientsReachEveryProjection() {
            float[] x = TrainParity.uniform(41, 2 * 4 * 8, -1f, 1f);
            float[] gOut = TrainParity.uniform(42, x.length, -1f, 1f);
            MultiHeadAttention mha = new MultiHeadAttention(8, 2, 0f, true, true);
            mha.initialize(host, new Random(SEED));
            Tensor in = leaf(host, x, 2, 4, 8);
            int before = graph.size();
            try (Tensor g = tensor(host, gOut, 2, 4, 8)) {
                try (Tensor out = mha.forward(in)) {
                    assertTrue("attention output is recorded", out.requiresGrad());
                    out.backward(g);
                }
                assertEquals("backward releases the attention graph", before, graph.size());
                assertEveryParameterHasGradient("mha", mha);
                assertNotNull("the input receives a gradient", in.grad());
                assertEquals(in.shape(), in.grad().shape());
                double[] biasGrad = new double[8];
                for (int i = 0; i < gOut.length; i++) {
                    biasGrad[i % 8] += gOut[i];
                }
                assertChain("out_proj bias gradient is the column sum of the upstream gradient", biasGrad,
                    mha.outProj().bias().grad().toFloatArray());
            } finally {
                closeLeaves(in);
                mha.close();
            }
        }

        @Test
        public void transformerBlocksRecordAndKeepTheirInputsOpen() {
            float[] x = TrainParity.uniform(51, 2 * 3 * 8, -1f, 1f);
            float[] m = TrainParity.uniform(52, 2 * 5 * 8, -1f, 1f);
            float[] gOut = TrainParity.uniform(53, x.length, -1f, 1f);
            TransformerEncoderLayer preLn = new TransformerEncoderLayer(8, 2, 16, 0f, true,
                TransformerEncoderLayer.NormType.LAYER_NORM, TransformerEncoderLayer.FFNType.MLP, false);
            TransformerEncoderLayer postRms = new TransformerEncoderLayer(8, 2, 16, 0f, false,
                TransformerEncoderLayer.NormType.RMS_NORM, TransformerEncoderLayer.FFNType.GEGLU, true);
            TransformerDecoderLayer decoder = new TransformerDecoderLayer(8, 2, 16, 0f, true,
                TransformerDecoderLayer.NormType.LAYER_NORM, TransformerDecoderLayer.FFNType.GLU);
            TransformerEncoder stack = new TransformerEncoder(2, 8, 2, 16, 0f, true,
                TransformerEncoderLayer.NormType.RMS_NORM, TransformerEncoderLayer.FFNType.MLP, false);
            SinusoidalPE pe = new SinusoidalPE(8, 6);
            Module[] blocks = {preLn, postRms, decoder, stack};
            Tensor in = leaf(host, x, 2, 3, 8);
            Tensor memory = leaf(host, m, 2, 5, 8);
            try (Tensor g = tensor(host, gOut, 2, 3, 8)) {
                Random random = new Random(SEED);
                preLn.initialize(host, random);
                postRms.initialize(host, random);
                decoder.initialize(host, random);
                stack.initialize(host, random);
                pe.initialize(host);
                stack.setPositionalEncoding(pe);
                for (Module block : blocks) {
                    String what = block.moduleName();
                    int before = graph.size();
                    try (Tensor out = block == decoder ? decoder.forward(in, memory) : block.forward(in)) {
                        assertEquals(what + " shape", in.shape(), out.shape());
                        assertTrue(what + " output is recorded", out.requiresGrad());
                        out.backward(g);
                    }
                    assertEquals(what + " backward releases the block's graph", before, graph.size());
                    assertEveryParameterHasGradient(what, block);
                    assertArrayEquals(what + " leaves the caller's input open", x, in.toFloatArray(), 0f);
                    block.zeroGrad();
                }
                assertNotNull("the encoder memory receives the cross-attention gradient", memory.grad());
                assertArrayEquals("the decoder leaves the memory open", m, memory.toFloatArray(), 0f);
            } finally {
                closeLeaves(in, memory);
                for (Module block : blocks) {
                    block.close();
                }
                pe.close();
            }
        }

        @Test
        public void learnedPositionTableReceivesSummedRowGradients() {
            float[] x = TrainParity.uniform(61, 2 * 4 * 8, -1f, 1f);
            float[] gOut = TrainParity.uniform(62, x.length, -1f, 1f);
            LearnedPE learned = new LearnedPE(8, 6);
            learned.initialize(host, new Random(SEED));
            int before = graph.size();
            try (Tensor in = tensor(host, x, 2, 4, 8); Tensor g = tensor(host, gOut, 2, 4, 8)) {
                try (Tensor out = learned.apply(in, 1)) {
                    assertTrue("the position table makes the result recorded", out.requiresGrad());
                    out.backward(g);
                }
                assertEquals(before, graph.size());
                Parameter table = learned.getPEEmbedding();
                assertNull("the table stays a leaf", table.tensor().gradFn());
                float[] grad = table.grad().toFloatArray();
                for (int row = 0; row < 6; row++) {
                    for (int c = 0; c < 8; c++) {
                        int s = row - 1;
                        float expected = s >= 0 && s < 4 ? gOut[s * 8 + c] + gOut[32 + s * 8 + c] : 0f;
                        assertEquals("table row " + row + " col " + c, expected, grad[row * 8 + c], 1e-6f);
                    }
                }
            } finally {
                learned.close();
            }
        }

        @Test
        public void learnedPositionEncodingIsATrainedChildOfTheStack() {
            float[] x = TrainParity.uniform(63, 2 * 4 * 8, -1f, 1f);
            float[] gOut = TrainParity.uniform(64, x.length, -1f, 1f);
            TransformerEncoder stack = new TransformerEncoder(1, 8, 2, 16, 0f, true,
                TransformerEncoderLayer.NormType.LAYER_NORM, TransformerEncoderLayer.FFNType.MLP, false);
            TransformerDecoder decoderStack = new TransformerDecoder(1, 8, 2);
            LearnedPE learned = new LearnedPE(8, 6);
            LearnedPE decoderPe = new LearnedPE(8, 6);
            try {
                Random random = new Random(SEED);
                stack.initialize(host, random);
                learned.initialize(host, random);
                decoderPe.initialize(host, random);
                stack.setPositionalEncoding(learned);
                Parameter table = learned.getPEEmbedding();
                assertTrue("the table is a stack parameter", containsIdentity(stack.parameters(), table));
                assertSame(table, stack.namedParameters().get("positional_encoding.pe_embedding"));
                assertTrue(stack.stateDict().containsKey("positional_encoding.pe_embedding"));
                stack.eval();
                assertFalse("eval() reaches the encoding", learned.isTraining());
                stack.train();
                assertTrue(learned.isTraining());

                SGD sgd = new SGD(stack.parameters(), 0.1f);
                float[] before = table.tensor().toFloatArray();
                int size = graph.size();
                try (Tensor in = tensor(host, x, 2, 4, 8); Tensor g = tensor(host, gOut, 2, 4, 8)) {
                    try (Tensor out = stack.forward(in);
                         Tensor weighted = TensorOps.mul(out, g);
                         Tensor loss = TensorOps.sum(weighted)) {
                        loss.backward();
                    }
                }
                assertEquals("backward releases the step's graph", size, graph.size());
                assertNotNull("the table receives a gradient", table.grad());
                sgd.step();
                assertFalse("the optimizer built from the stack's parameters updates the table",
                    Arrays.equals(before, table.tensor().toFloatArray()));
                sgd.zeroGrad();

                stack.setPositionalEncoding(null);
                assertFalse("removing the encoding unregisters it", containsIdentity(stack.parameters(), table));
                assertNull(stack.namedParameters().get("positional_encoding.pe_embedding"));
                assertEquals("the stack hands the encoding back unclosed", 48, table.tensor().numElements());

                decoderStack.setPositionalEncoding(decoderPe);
                assertSame(decoderPe.getPEEmbedding(),
                    decoderStack.namedParameters().get("positional_encoding.pe_embedding"));
            } finally {
                stack.close();
                decoderStack.close();
                learned.close();
                decoderPe.close();
            }
        }

        @Test
        public void kvCacheAppendIsDetachedAndRecordsNothing() {
            float[] data = TrainParity.uniform(71, 2 * 2 * 3, -1f, 1f);
            Tensor keys = leaf(host, data, 1, 2, 2, 3);
            KVCache cache = new KVCache(1, 2, 3, 4, 1);
            try {
                cache.initialize(host);
                int before = graph.size();
                cache.append(0, keys, keys);
                assertEquals("a cache write records nothing", before, graph.size());
                try (Tensor cached = cache.getKeys(0)) {
                    assertFalse("cached keys are constants", cached.requiresGrad());
                    assertArrayEquals(data, cached.toFloatArray(), 0f);
                }
                cache.clear();
                assertEquals(before, graph.size());
                assertNull(keys.grad());
            } finally {
                cache.close();
                closeLeaves(keys);
            }
        }

        @Test
        public void mixtureOfExpertsRoutesGradientsAndHandsOverTheLoadBalanceTerm() {
            float[] x = TrainParity.uniform(81, 2 * 3 * 6, -1f, 1f);
            float[] gOut = TrainParity.uniform(82, x.length, -1f, 1f);
            MixtureOfExperts moe = new MixtureOfExperts(6, 10, 4, 2);
            moe.initialize(host, new Random(SEED));
            int before = graph.size();
            try (Tensor in = tensor(host, x, 2, 3, 6); Tensor g = tensor(host, gOut, 2, 3, 6)) {
                MoeStep step = MoeStep.run(moe, in, g);
                assertEquals("one backward releases the routed graph and the auxiliary term", before, graph.size());
                assertTrue("the auxiliary term is recorded", step.auxRequiresGrad);
                assertEquals("the auxiliary term is the load-balance loss", step.loadBalance, step.auxValue, 0f);
                assertEquals("the detached load-balance handle survives backward", step.loadBalance,
                    moe.getLoadBalanceLoss(), 0f);
                assertArrayEquals("the detached router probabilities survive backward", step.routerProbs,
                    moe.getLastRouterProbs(), 0f);
                assertNotNull("the gates carry the router's gradient", moe.router().weight().grad());
                assertNotNull(moe.router().bias().grad());
                int[] indices = moe.getLastExpertIndices();
                for (int e = 0; e < moe.numExperts(); e++) {
                    boolean routed = false;
                    for (int index : indices) {
                        routed |= index == e;
                    }
                    for (Parameter p : moe.experts().get(e).parameters()) {
                        assertEquals("expert " + e + " has a gradient exactly when tokens were routed to it", routed,
                            p.grad() != null);
                    }
                }
            } finally {
                moe.close();
            }
        }

        @Test
        public void weightSharedMixtureOfExpertsAddsTheLoadBalanceTermOfEachApplication() {
            float[] x = TrainParity.uniform(83, 2 * 3 * 6, -1f, 1f);
            float[] gOut = TrainParity.uniform(84, x.length, -1f, 1f);
            MixtureOfExperts moe = new MixtureOfExperts(6, 10, 4, 2);
            moe.initialize(host, new Random(SEED));
            int before = graph.size();
            try (Tensor in = tensor(host, x, 2, 3, 6); Tensor g = tensor(host, gOut, 2, 3, 6)) {
                try (Tensor first = moe.forward(in)) {
                    float firstTerm = moe.getLoadBalanceLoss();
                    try (Tensor second = moe.forward(first)) {
                        float secondTerm = moe.getLoadBalanceLoss();
                        try (Tensor aux = moe.takeAuxiliaryLoss()) {
                            assertNull("both terms are handed over at once", moe.takeAuxiliaryLoss());
                            assertTrue("the sum of the terms is recorded", aux.requiresGrad());
                            assertEquals("the loss holds the term of each application", firstTerm + secondTerm,
                                aux.toFloatArray()[0], 0f);
                            try (Tensor weighted = TensorOps.mul(second, g);
                                 Tensor main = TensorOps.sum(weighted);
                                 Tensor total = TensorOps.add(main, aux)) {
                                total.backward();
                            }
                        }
                    }
                }
                assertEquals("one backward releases both applications and both terms", before, graph.size());
                assertNotNull(moe.router().weight().grad());
                assertNotNull(moe.router().bias().grad());
            } finally {
                moe.close();
            }
        }

        @Test
        public void mixtureOfExpertsAddsATermOnlyInTrainingMode() {
            float[] x = TrainParity.uniform(85, 2 * 3 * 6, -1f, 1f);
            MixtureOfExperts moe = new MixtureOfExperts(6, 10, 4, 2);
            moe.initialize(host, new Random(SEED));
            int before = graph.size();
            try (Tensor in = tensor(host, x, 2, 3, 6);
                 ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                moe.eval();
                try (Tensor out = moe.forward(in)) {
                    assertNull("eval mode adds no auxiliary term", moe.takeAuxiliaryLoss());
                }
                float evalTerm = moe.getLoadBalanceLoss();
                moe.train();
                try (Tensor out = moe.forward(in); Tensor aux = moe.takeAuxiliaryLoss()) {
                    assertNotNull("training mode adds the term", aux);
                    assertFalse("a term computed under no-grad is a constant", aux.requiresGrad());
                    assertEquals(evalTerm, aux.toFloatArray()[0], 0f);
                }
            } finally {
                moe.close();
            }
            assertEquals(before, graph.size());
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
            ComputationGraph.setGradEnabled(true);
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
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

        @Test
        public void flashAttentionGradientsMatchCpuPath() {
            for (Scenario sc : new Scenario[]{new Scenario(3, 4, false), new Scenario(4, 4, true)}) {
                String what = api + (sc.causal ? " causal" : " masked");
                float[] gOut = TrainParity.uniform(91, sc.q.length, -1f, 1f);
                Tensor[] hLeaves = attentionLeaves(host, sc);
                Tensor[] gLeaves = attentionLeaves(gpu, sc);
                try (Tensor hm = sc.mask(host); Tensor gm = sc.mask(gpu);
                     Tensor hg = tensor(host, gOut, sc.b, sc.h, sc.sq, sc.d);
                     Tensor gg = tensor(gpu, gOut, sc.b, sc.h, sc.sq, sc.d);
                     FlashAttentionOp.FlashAttentionResult hr =
                         FlashAttentionOp.apply(hLeaves[0], hLeaves[1], hLeaves[2], sc.scale, sc.causal, hm);
                     FlashAttentionOp.FlashAttentionResult gr =
                         FlashAttentionOp.apply(gLeaves[0], gLeaves[1], gLeaves[2], sc.scale, sc.causal, gm)) {
                    hr.output.backward(hg);
                    gr.output.backward(gg);
                    for (int i = 0; i < 3; i++) {
                        assertSame(gpu, gLeaves[i].grad().backend());
                        assertChain(what + " grad " + i, toDouble(hLeaves[i].grad().toFloatArray()),
                            gLeaves[i].grad().toFloatArray());
                    }
                } finally {
                    closeLeaves(hLeaves);
                    closeLeaves(gLeaves);
                }
            }
        }

        @Test
        public void multiHeadAttentionGradientsMatchCpuPath() {
            float[] x = TrainParity.uniform(101, 2 * 5 * 8, -1f, 1f);
            float[] kv = TrainParity.uniform(102, 2 * 3 * 8, -1f, 1f);
            float[] gOut = TrainParity.uniform(103, x.length, -1f, 1f);
            float[] keyBias = {0f, Float.NEGATIVE_INFINITY, 0.5f, -0.25f, 0f, 0f};
            MultiHeadAttention ref = new MultiHeadAttention(8, 2, 0f, true, true);
            MultiHeadAttention dev = new MultiHeadAttention(8, 2, 0f, true, true);
            ref.initialize(host, new Random(SEED));
            dev.initialize(gpu, new Random(SEED));
            dev.loadStateDict(ref.stateDict());
            Tensor hx = leaf(host, x, 2, 5, 8);
            Tensor gx = leaf(gpu, x, 2, 5, 8);
            Tensor hkv = leaf(host, kv, 2, 3, 8);
            Tensor gkv = leaf(gpu, kv, 2, 3, 8);
            try (Tensor hm = tensor(host, keyBias, 2, 1, 1, 3); Tensor gm = tensor(gpu, keyBias, 2, 1, 1, 3);
                 Tensor hg = tensor(host, gOut, 2, 5, 8); Tensor gg = tensor(gpu, gOut, 2, 5, 8)) {
                try (Tensor ho = ref.forward(hx, hkv, hkv, hm); Tensor go = dev.forward(gx, gkv, gkv, gm)) {
                    ho.backward(hg);
                    go.backward(gg);
                }
                assertSame(gpu, gx.grad().backend());
                assertChain(api + " query input grad", toDouble(hx.grad().toFloatArray()), gx.grad().toFloatArray());
                assertChain(api + " key/value input grad", toDouble(hkv.grad().toFloatArray()),
                    gkv.grad().toFloatArray());
                assertParameterGradientsMatch(api + " mha", ref, dev);
            } finally {
                closeLeaves(hx, gx, hkv, gkv);
                ref.close();
                dev.close();
            }
        }

        @Test
        public void mixtureOfExpertsGradientsMatchCpuPath() {
            MixtureOfExperts ref = new MixtureOfExperts(6, 10, 4, 2);
            MixtureOfExperts dev = new MixtureOfExperts(6, 10, 4, 2);
            ref.initialize(host, new Random(SEED));
            dev.initialize(gpu, new Random(SEED));
            dev.loadStateDict(ref.stateDict());
            float[] x = TrainParity.uniform(111, 3 * 4 * 6, -1f, 1f);
            float[] gOut = TrainParity.uniform(112, x.length, -1f, 1f);
            try (Tensor hx = tensor(host, x, 3, 4, 6); Tensor gx = tensor(gpu, x, 3, 4, 6);
                 Tensor hg = tensor(host, gOut, 3, 4, 6); Tensor gg = tensor(gpu, gOut, 3, 4, 6)) {
                MoeStep.run(ref, hx, hg);
                MoeStep.run(dev, gx, gg);
                assertArrayEquals(api + " expert indices", ref.getLastExpertIndices(), dev.getLastExpertIndices());
                assertParameterGradientsMatch(api + " moe", ref, dev);
            } finally {
                ref.close();
                dev.close();
            }
        }
    }

    // ------------------------------------------------------------------ shared steps and helpers

    /** One mixture-of-experts training step: loss {@code sum(out * g) + auxiliary term}, one backward pass. */
    static final class MoeStep {
        boolean auxRequiresGrad;
        float auxValue;
        float loadBalance;
        float[] routerProbs;

        static MoeStep run(MixtureOfExperts moe, Tensor in, Tensor g) {
            MoeStep step = new MoeStep();
            try (Tensor out = moe.forward(in);
                 Tensor aux = moe.takeAuxiliaryLoss()) {
                assertNotNull("load balancing hands over an auxiliary term", aux);
                assertNull("the term is handed over once", moe.takeAuxiliaryLoss());
                step.auxRequiresGrad = aux.requiresGrad();
                step.auxValue = aux.toFloatArray()[0];
                step.loadBalance = moe.getLoadBalanceLoss();
                step.routerProbs = moe.getLastRouterProbs();
                try (Tensor weighted = TensorOps.mul(out, g);
                     Tensor main = TensorOps.sum(weighted);
                     Tensor total = TensorOps.add(main, aux)) {
                    total.backward();
                }
            }
            return step;
        }
    }

    /** Query, key and value leaves of {@code sc} on {@code be}. */
    static Tensor[] attentionLeaves(ComputeBackend be, Scenario sc) {
        Tensor[] leaves = new Tensor[3];
        try {
            leaves[0] = leaf(be, sc.q, sc.b, sc.h, sc.sq, sc.d);
            leaves[1] = leaf(be, sc.k, sc.b, sc.h, sc.skv, sc.d);
            leaves[2] = leaf(be, sc.v, sc.b, sc.h, sc.skv, sc.d);
            return leaves;
        } catch (RuntimeException e) {
            closeLeaves(leaves);
            throw e;
        }
    }

    /** A leaf with {@code data} in {@code dims} on {@code be}. */
    static Tensor leaf(ComputeBackend be, float[] data, int... dims) {
        try (Tensor t = tensor(be, data, dims)) {
            return t.withRequiresGrad(true);
        }
    }

    /** Closes each leaf's gradient and the leaf; nulls are skipped. */
    static void closeLeaves(Tensor... leaves) {
        for (Tensor t : leaves) {
            if (t != null) {
                if (t.grad() != null) {
                    t.grad().close();
                }
                t.close();
            }
        }
    }

    static boolean containsIdentity(Iterable<Parameter> params, Parameter p) {
        for (Parameter candidate : params) {
            if (candidate == p) {
                return true;
            }
        }
        return false;
    }

    static void assertEveryParameterHasGradient(String what, Module module) {
        for (Map.Entry<String, Parameter> entry : module.namedParameters().entrySet()) {
            Parameter p = entry.getValue();
            assertNotNull(what + " " + entry.getKey() + " has a gradient", p.grad());
            assertEquals(what + " " + entry.getKey() + " gradient shape", p.shape(), p.grad().shape());
        }
    }

    /** Every parameter gradient of {@code dev} matches {@code ref}'s, and is null exactly where {@code ref}'s is. */
    static void assertParameterGradientsMatch(String what, Module ref, Module dev) {
        Map<String, Parameter> devParams = dev.namedParameters();
        for (Map.Entry<String, Parameter> entry : ref.namedParameters().entrySet()) {
            String name = what + " " + entry.getKey();
            Tensor refGrad = entry.getValue().grad();
            Tensor devGrad = devParams.get(entry.getKey()).grad();
            if (refGrad == null) {
                assertNull(name + " has no gradient on either path", devGrad);
                continue;
            }
            assertNotNull(name + " gradient", devGrad);
            assertEquals(name + " gradient shape", refGrad.shape(), devGrad.shape());
            assertChain(name + " gradient", toDouble(refGrad.toFloatArray()), devGrad.toFloatArray());
        }
    }
}
