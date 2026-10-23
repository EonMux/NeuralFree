package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.IndexGradFunctions;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The nn/graph modules on their CPU path and on each GPU API, each against a double-precision reference of the module
 * formula. The modules chain at most three reductions of length at most 8 over values of magnitude about 1, with
 * transcendentals in between (the reduction and transcendental classes of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}), so both paths are held to
 * {@code |out - ref| <= 1e-4 * max(1, |ref|)}. Structure-only results are exact: the CSR, the clamped gathers and the
 * aggregation of small integers, the global max pool (bit-identical to the CPU path) and the top-k selection.
 *
 * <p>The forward comparisons run under {@link ComputationGraph#noGrad()} and check that nothing is recorded. The
 * autograd checks run in grad mode: every module's forward records a graph that reaches each of its parameters, the
 * backward of a summed output leaves the thread's graph at its previous size, GAT's softmax shift is not recorded, a
 * rejected graph records nothing, and the gradients of the clamped gather, the segment sum and the global pools match
 * exact values, and TopKPool's weight and input gradients under each gate (through the norm of its projection) match
 * a double-precision reference. On each GPU API the parameter gradients are compared with the CPU path's. Parameters
 * are written under {@code noGrad}, as the in-place guard requires.</p>
 */
@RunWith(Enclosed.class)
public class GraphModulesParityTest {

    static final double TOL = 1e-4;
    static final int N = 6;
    static final int F = 4;
    /** Edges {@code SRC[e] -> DST[e]}: a duplicate, a self loop, and nodes 4 and 5 without incoming edges. */
    static final int[] SRC = {0, 2, 3, 1, 4, 5, 2, 0, 5, 0};
    static final int[] DST = {1, 1, 1, 0, 0, 2, 2, 3, 3, 1};
    /** The TopKPool projection of every scenario: it keeps the scores of the scenario features well separated. */
    static final float[] TOP_K_WEIGHT = {1.0f, 0.5f, -0.25f, 2.0f};

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
        public void convolutionsMatchReferences() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Scenarios.all(host, "cpu path");
            }
        }

        @Test
        public void poolsMatchReferences() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Scenarios.pools(host, "cpu path");
            }
        }

        @Test
        public void forwardRecordsAndReachesEveryParameter() {
            Gradients.parameterGradients(host, "cpu path");
        }

        @Test
        public void gradientsOfGatherSegmentAndPoolsAreExact() {
            Gradients.exact(host, "cpu path");
        }

        @Test
        public void gatSoftmaxShiftIsNotRecorded() {
            Gradients.gatShiftDetached(host, "cpu path");
        }

        @Test
        public void topKGateGradientsMatchReference() {
            Gradients.topKGate(host, "cpu path");
        }

        @Test
        public void edgelessForwardLeavesNoStrayNode() {
            Gradients.edgelessRecording(host, "cpu path");
        }

        @Test
        public void csrIsTargetSortedStableAndCached() {
            try (GraphData g = Scenarios.graph(host, SRC, DST)) {
                assertArrayEquals("offsets", new int[]{0, 2, 6, 8, 10, 10, 10}, g.csrOffsets().toIntArray());
                assertArrayEquals("edges", new int[]{3, 4, 0, 1, 2, 9, 5, 6, 7, 8}, g.csrEdges().toIntArray());
                assertArrayEquals("in-degree", new int[]{2, 4, 2, 2, 0, 0}, g.inDegree().toIntArray());
                assertSame("offsets cached", g.csrOffsets(), g.csrOffsets());
                assertSame("edges cached", g.csrEdges(), g.csrEdges());
            }
        }

        @Test
        public void outOfRangeIndicesFollowTheCpuRules() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Scenarios.clampedSourcesAndIgnoredTargets(host, "cpu path");
            }
            Scenarios.rejectedIndices(host);
        }

        @Test
        public void edgelessGraphsStillApplyTheUpdate() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Scenarios.edgeless(host, "cpu path");
            }
        }

        @Test
        public void gatAttentionDropoutAppliesOnlyInTraining() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Settings.gatAttentionDropout(host, "cpu path");
            }
            Settings.gatDropoutRecording(host, "cpu path");
        }

        @Test
        public void topKMinScoreKeepsNodesAboveTheGraphThreshold() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Settings.topKMinScore(host, "cpu path");
            }
        }

        @Test
        public void invalidSettingsAreRejected() {
            Settings.rejectedSettings();
        }
    }

    /** The device path on each GPU API, against the same references. */
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
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                TrainGpu.release(host);
            } finally {
                if (session != null) {
                    session.close();
                }
            }
        }

        @Test
        public void convolutionsMatchReferences() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Scenarios.all(gpu, api.toString());
            }
        }

        @Test
        public void poolsMatchCpuPath() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                float[][] cpu = Scenarios.pools(host, "cpu path");
                float[][] dev = Scenarios.pools(gpu, api.toString());
                TrainParity.assertDivSqrt(api + " mean pool vs cpu path", cpu[0], dev[0]);
                TrainParity.assertBitIdentical(api + " max pool vs cpu path", cpu[1], dev[1]);
                assertArrayEquals(api + " top-k selection vs cpu path", cpu[2], dev[2], 0.0f);
            }
        }

        @Test
        public void parameterGradientsMatchCpuPath() {
            List<float[]> cpu = Gradients.parameterGradients(host, "cpu path");
            List<float[]> dev = Gradients.parameterGradients(gpu, api.toString());
            assertEquals(api + " gradient count", cpu.size(), dev.size());
            for (int i = 0; i < cpu.size(); i++) {
                Gradients.assertGradientClose(api + " parameter gradient " + i + " vs cpu path", cpu.get(i),
                    dev.get(i));
            }
        }

        @Test
        public void gradientsOfGatherSegmentAndPoolsAreExact() {
            Gradients.exact(gpu, api.toString());
        }

        @Test
        public void gatSoftmaxShiftIsNotRecorded() {
            Gradients.gatShiftDetached(gpu, api.toString());
        }

        @Test
        public void topKGateGradientsMatchReference() {
            Gradients.topKGate(gpu, api.toString());
        }

        @Test
        public void edgelessForwardLeavesNoStrayNode() {
            Gradients.edgelessRecording(gpu, api.toString());
        }

        @Test
        public void outOfRangeIndicesFollowTheCpuRules() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Scenarios.clampedSourcesAndIgnoredTargets(gpu, api.toString());
            }
            Scenarios.rejectedIndices(gpu);
        }

        @Test
        public void edgelessGraphsStillApplyTheUpdate() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Scenarios.edgeless(gpu, api.toString());
            }
        }

        @Test
        public void gatAttentionDropoutAppliesOnlyInTraining() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Settings.gatAttentionDropout(gpu, api.toString());
            }
            Settings.gatDropoutRecording(gpu, api.toString());
        }

        @Test
        public void topKMinScoreKeepsNodesAboveTheGraphThreshold() {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Settings.topKMinScore(gpu, api.toString());
            }
        }
    }

    // ------------------------------------------------------------------ scenarios

    /** Module runs on one backend, checked against the double references. */
    static final class Scenarios {
        private Scenarios() {
        }

        static GraphData graph(ComputeBackend be, int[] src, int[] dst) {
            return graph(be, TrainParity.uniform(7, N * F, -1f, 1f), N, src, dst, null);
        }

        static GraphData graph(ComputeBackend be, float[] x, int nodes, int[] src, int[] dst, float[] batch) {
            int[] index = new int[2 * src.length];
            System.arraycopy(src, 0, index, 0, src.length);
            System.arraycopy(dst, 0, index, src.length, dst.length);
            Tensor nodeFeatures;
            Tensor edgeIndex;
            try (Tensor flat = Tensor.fromArray(be, x)) {
                nodeFeatures = flat.reshape(nodes, x.length / nodes);
            }
            try (Tensor flat = Tensor.fromArray(be, index)) {
                edgeIndex = flat.reshape(2, src.length);
            }
            return new GraphData(nodeFeatures, edgeIndex, null, batch == null ? null : Tensor.fromArray(be, batch));
        }

        static void all(ComputeBackend be, String what) {
            gcn(be, what, true);
            gcn(be, what, false);
            gat(be, what, true);
            gat(be, what, false);
            for (GraphSAGEConv.Aggregator a : GraphSAGEConv.Aggregator.values()) {
                sage(be, what, a);
            }
            gin(be, what, false);
            gin(be, what, true);
        }

        static void gcn(ComputeBackend be, String what, boolean normalize) {
            GCNConv m = new GCNConv(F, 3, true, normalize);
            m.initialize(be, new Random(11));
            try (GraphData g = graph(be, SRC, DST)) {
                float[] bias = {0.5f, -0.25f, 1f};
                store(m.bias(), bias);
                double[] ref = Reference.gcn(g.nodeFeatures().toFloatArray(), SRC, DST,
                    m.weight().tensor().toFloatArray(), 3, bias, normalize);
                assertClose(what + " GCNConv normalize=" + normalize, ref, run(m.forward(g)));
            } finally {
                m.close();
            }
        }

        static void gat(ComputeBackend be, String what, boolean concat) {
            int heads = 2;
            int out = 3;
            GATConv m = new GATConv(F, out, heads, concat, true, 0.2f, 0.0f);
            m.initialize(be, new Random(12));
            try (GraphData g = graph(be, SRC, DST)) {
                float[] bias = TrainParity.uniform(13, concat ? heads * out : out, -1f, 1f);
                store(m.bias(), bias);
                double[] ref = Reference.gat(g.nodeFeatures().toFloatArray(), SRC, DST,
                    m.weight().tensor().toFloatArray(), m.attSrc().tensor().toFloatArray(),
                    m.attDst().tensor().toFloatArray(), heads, out, 0.2f, concat, bias);
                assertClose(what + " GATConv concat=" + concat, ref, run(m.forward(g)));
            } finally {
                m.close();
            }
        }

        static void sage(ComputeBackend be, String what, GraphSAGEConv.Aggregator aggregator) {
            for (boolean normalize : new boolean[]{false, true}) {
                GraphSAGEConv m = new GraphSAGEConv(F, 3, aggregator, true, normalize);
                m.initialize(be, new Random(14));
                try (GraphData g = graph(be, SRC, DST)) {
                    float[] bias = {0.1f, -0.2f, 0.3f};
                    store(m.bias(), bias);
                    double[] ref = Reference.sage(g.nodeFeatures().toFloatArray(), SRC, DST,
                        m.weightSelf().tensor().toFloatArray(), m.weightNeigh().tensor().toFloatArray(), 3, bias,
                        aggregator == GraphSAGEConv.Aggregator.MAX_POOL, normalize);
                    assertClose(what + " GraphSAGEConv " + aggregator + " normalize=" + normalize, ref,
                        run(m.forward(g)));
                } finally {
                    m.close();
                }
            }
        }

        static void gin(ComputeBackend be, String what, boolean trainEps) {
            GINConv m = new GINConv(F, 3, 0.5f, trainEps);
            m.initialize(be, new Random(15));
            try (GraphData g = graph(be, SRC, DST)) {
                float eps = 0.5f;
                if (trainEps) {
                    eps = 0.25f;
                    store(m.getParameter("eps"), new float[]{eps});
                }
                double[] ref = Reference.gin(g.nodeFeatures().toFloatArray(), SRC, DST, eps,
                    floats(m.mlp(), "weight1"), floats(m.mlp(), "bias1"), floats(m.mlp(), "weight2"),
                    floats(m.mlp(), "bias2"), 3);
                assertClose(what + " GINConv trainEps=" + trainEps, ref, run(m.forward(g)));
            } finally {
                m.close();
            }
        }

        /** {mean pool, max pool, top-k selected node ids} of a batch of three graphs (graph 1 empty). */
        static float[][] pools(ComputeBackend be, String what) {
            float[] x = TrainParity.uniform(16, N * F, -1f, 1f);
            float[] batch = {0, 2, 0, 2, 2, 0};
            float[] mean;
            float[] max;
            try (GraphData g = graph(be, x, N, SRC, DST, batch)) {
                assertEquals(what + " numGraphs", 3, g.numGraphs());
                mean = run(new GlobalMeanPool().forward(g));
                max = run(new GlobalMaxPool().forward(g));
                assertClose(what + " GlobalMeanPool", Reference.pool(x, batch, 3, false), mean);
                assertClose(what + " GlobalMaxPool", Reference.pool(x, batch, 3, true), max);
            }
            try (GraphData g = graph(be, x, N, SRC, DST, null)) {
                assertClose(what + " GlobalMeanPool unbatched", Reference.pool(x, null, 1, false),
                    run(new GlobalMeanPool().forward(g)));
                assertClose(what + " GlobalMaxPool unbatched", Reference.pool(x, null, 1, true),
                    run(new GlobalMaxPool().forward(g)));
            }
            return new float[][]{mean, max, topK(be, what, x, batch)};
        }

        /**
         * TopKPool with weights that keep the scores well separated, under each gate, on the unbatched graph (3 of 6
         * nodes) and on the batch of {@link #pools} (2 of the 3 nodes of graphs 0 and 2, none of the empty graph 1);
         * returns the selected node ids of every run.
         */
        private static float[] topK(ComputeBackend be, String what, float[] x, float[] batch) {
            float[] w = TOP_K_WEIGHT;
            List<Integer> ids = new ArrayList<>();
            for (TopKPool.Nonlinearity act : TopKPool.Nonlinearity.values()) {
                String label = what + " TopKPool " + act;
                TopKPool m = new TopKPool(F, 0.5f, act);
                m.initialize(be, new Random(17));
                store(m.weight(), w);
                try (GraphData whole = graph(be, x, N, SRC, DST, null);
                     GraphData batched = graph(be, x, N, SRC, DST, batch)) {
                    int[] single = checkTopK(label, m, whole, Reference.topK(x, w, null, 1, 0.5f, act));
                    assertEquals(label + " keeps 3 of 6 nodes", 3, single.length);
                    int[] perGraph = checkTopK(label + " per graph", m, batched,
                        Reference.topK(x, w, batch, 3, 0.5f, act));
                    assertEquals(label + " keeps 2 of 3 nodes per non-empty graph", 4, perGraph.length);
                    for (int id : single) {
                        ids.add(id);
                    }
                    for (int id : perGraph) {
                        ids.add(id);
                    }
                } finally {
                    m.close();
                }
            }
            float[] selected = new float[ids.size()];
            for (int i = 0; i < selected.length; i++) {
                selected[i] = ids.get(i);
            }
            return selected;
        }

        /** Checks the selection, the pooled features and every node's gated score against {@code ref}. */
        private static int[] checkTopK(String what, TopKPool m, GraphData g, double[][] ref) {
            TopKPool.TopKPoolResult r = m.forward(g);
            int[] expected = new int[ref[1].length];
            for (int i = 0; i < expected.length; i++) {
                expected[i] = (int) ref[1][i];
            }
            assertArrayEquals(what + " selection", expected, r.selectedIndices());
            assertClose(what + " features", ref[0], run(r.pooledFeatures()));
            assertClose(what + " scores", ref[2], r.scores());
            return r.selectedIndices();
        }

        /**
         * GCN without normalization or bias and with the identity weight returns the raw sum aggregation: sources
         * -3 and 9 clamp to rows 0 and 3, targets -1 and 4 are ignored. The inputs are small integers, so the sums
         * are exact.
         */
        static void clampedSourcesAndIgnoredTargets(ComputeBackend be, String what) {
            float[] x = {1, 2, 10, 20, 100, 200, 1000, 2000};
            int[] src = {-3, 9, 1, 2, 0};
            int[] dst = {1, 1, -1, 4, 2};
            GCNConv m = new GCNConv(2, 2, false, false);
            m.initialize(be, new Random(18));
            try (GraphData g = graph(be, x, 4, src, dst, null)) {
                store(m.weight(), new float[]{1, 0, 0, 1});
                float[] expected = {0, 0, 1001, 2002, 1, 2, 0, 0};
                TrainParity.assertBitIdentical(what + " clamped gather", expected, run(m.forward(g)));
            } finally {
                m.close();
            }
        }

        /**
         * GAT rejects an out-of-range target and normalized GCN any out-of-range index, in grad mode before any
         * computation: nothing is recorded.
         */
        static void rejectedIndices(ComputeBackend be) {
            GATConv gat = new GATConv(F, 2);
            gat.initialize(be, new Random(19));
            GCNConv gcn = new GCNConv(F, 2);
            gcn.initialize(be, new Random(20));
            try (GraphData badTarget = graph(be, new int[]{0, 1}, new int[]{1, N});
                 GraphData badSource = graph(be, new int[]{-1, 1}, new int[]{1, 0})) {
                ComputationGraph recorder = ComputationGraph.current();
                int before = recorder.size();
                expectOutOfRange("GATConv target", () -> gat.forward(badTarget));
                expectOutOfRange("GCNConv source", () -> gcn.forward(badSource));
                assertEquals("rejected graphs record nothing", before, recorder.size());
            } finally {
                gat.close();
                gcn.close();
            }
        }

        /** Without edges every node aggregates to zeros, so the output rows are the bias. */
        static void edgeless(ComputeBackend be, String what) {
            float[] bias = {1, -2, 3};
            GCNConv gcn = new GCNConv(F, 3);
            gcn.initialize(be, new Random(21));
            GATConv gat = new GATConv(F, 3, 1, true);
            gat.initialize(be, new Random(22));
            try (GraphData g = graph(be, new int[0], new int[0])) {
                store(gcn.bias(), bias);
                store(gat.bias(), bias);
                float[] expected = new float[N * 3];
                for (int n = 0; n < N; n++) {
                    System.arraycopy(bias, 0, expected, n * 3, 3);
                }
                TrainParity.assertBitIdentical(what + " edgeless GCNConv", expected, run(gcn.forward(g)));
                TrainParity.assertBitIdentical(what + " edgeless GATConv", expected, run(gat.forward(g)));
            } finally {
                gcn.close();
                gat.close();
            }
        }

        private static void expectOutOfRange(String what, Runnable call) {
            try {
                call.run();
                fail(what + ": expected IndexOutOfBoundsException");
            } catch (IndexOutOfBoundsException expected) {
                // the documented rejection
            }
        }

        private static float[] floats(Module m, String parameter) {
            return m.getParameter(parameter).tensor().toFloatArray();
        }

        /** Writes {@code values} into the parameter's tensor under {@code noGrad}, as the in-place guard requires. */
        static void store(Parameter p, float[] values) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                GpuOptimizerOps.storeFloats(p.tensor(), values);
            }
        }

        /** The host copy of {@code t}, a forward result computed under {@code noGrad}, which is closed. */
        private static float[] run(Tensor t) {
            try {
                assertFalse("a forward under noGrad must not require grad", t.requiresGrad());
                return t.toFloatArray();
            } finally {
                t.close();
            }
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

    // ------------------------------------------------------------------ autograd

    /** Grad-mode scenarios: recording, parameter reach, graph lifetime and exact gradients. */
    static final class Gradients {
        /** Node features of the gradient scenarios: the TopKPool scores of {@link Scenarios#pools} stay separated. */
        private static final long FEATURE_SEED = 16;

        private Gradients() {
        }

        /** One module configuration: how to build it and which tensor of its forward is summed into the loss. */
        private static final class Case {
            final String name;
            final Function<ComputeBackend, Module> build;
            final BiFunction<Module, GraphData, Tensor> forward;

            Case(String name, Function<ComputeBackend, Module> build, BiFunction<Module, GraphData, Tensor> forward) {
                this.name = name;
                this.build = build;
                this.forward = forward;
            }
        }

        private static List<Case> cases() {
            List<Case> cases = new ArrayList<>();
            for (boolean normalize : new boolean[]{true, false}) {
                cases.add(new Case("GCNConv normalize=" + normalize, be -> {
                    GCNConv m = new GCNConv(F, 3, true, normalize);
                    m.initialize(be, new Random(31));
                    return m;
                }, (m, g) -> ((GCNConv) m).forward(g)));
            }
            for (boolean concat : new boolean[]{true, false}) {
                cases.add(new Case("GATConv concat=" + concat, be -> {
                    GATConv m = new GATConv(F, 3, 2, concat, true, 0.2f, 0.0f);
                    m.initialize(be, new Random(32));
                    return m;
                }, (m, g) -> ((GATConv) m).forward(g)));
            }
            for (GraphSAGEConv.Aggregator a : GraphSAGEConv.Aggregator.values()) {
                cases.add(new Case("GraphSAGEConv " + a, be -> {
                    GraphSAGEConv m = new GraphSAGEConv(F, 3, a, true, true);
                    m.initialize(be, new Random(33));
                    return m;
                }, (m, g) -> ((GraphSAGEConv) m).forward(g)));
            }
            for (boolean trainEps : new boolean[]{true, false}) {
                cases.add(new Case("GINConv trainEps=" + trainEps, be -> {
                    GINConv m = new GINConv(F, 3, 0.5f, trainEps);
                    m.initialize(be, new Random(34));
                    return m;
                }, (m, g) -> ((GINConv) m).forward(g)));
            }
            for (TopKPool.Nonlinearity act : TopKPool.Nonlinearity.values()) {
                cases.add(new Case("TopKPool " + act, be -> {
                    TopKPool m = new TopKPool(F, 0.5f, act);
                    m.initialize(be, new Random(35));
                    Scenarios.store(m.weight(), TOP_K_WEIGHT);
                    return m;
                }, (m, g) -> ((TopKPool) m).forward(g).pooledFeatures()));
            }
            cases.add(new Case("GATConv attention dropout=0.5", be -> {
                GATConv m = new GATConv(F, 3, 2, true, true, 0.2f, 0.5f);
                m.initialize(be, new Random(32));
                return m;
            }, (m, g) -> ((GATConv) m).forward(g)));
            cases.add(new Case("TopKPool minScore=0.2", be -> {
                TopKPool m = new TopKPool(F, 0.5f, 0.2f);
                m.initialize(be, new Random(35));
                Scenarios.store(m.weight(), TOP_K_WEIGHT);
                return m;
            }, (m, g) -> ((TopKPool) m).forward(g).pooledFeatures()));
            return cases;
        }

        /**
         * For every case, in grad mode: the forward output requires grad, {@code sum(output).backward()} gives every
         * parameter a finite gradient of its shape, and the thread's graph is back at its size before the forward.
         * Returns the gradients in case and parameter order.
         */
        static List<float[]> parameterGradients(ComputeBackend be, String what) {
            List<float[]> gradients = new ArrayList<>();
            ComputationGraph recorder = ComputationGraph.current();
            for (Case c : cases()) {
                String label = what + " " + c.name;
                Module m = c.build.apply(be);
                try (GraphData g = Scenarios.graph(be, features(), N, SRC, DST, null)) {
                    int before = recorder.size();
                    backwardOfSum(label, c.forward.apply(m, g));
                    assertEquals(label + ": graph size after backward", before, recorder.size());
                    for (Map.Entry<String, Parameter> e : m.namedParameters().entrySet()) {
                        gradients.add(finiteGradient(label + " " + e.getKey(), e.getValue()));
                    }
                } finally {
                    m.close();
                }
            }
            return gradients;
        }

        /**
         * Exact gradients of the node features. GCN without normalization or bias and with the identity weight
         * (the graph of {@link Scenarios#clampedSourcesAndIgnoredTargets}): each node receives one per listed edge
         * whose clamped source it is (sources -3 and 0 give node 0 two, source 9 gives node 3 one, the edges with
         * targets -1 and 4 give nothing), and the weight receives the column sums of the aggregate. Mean pool: each
         * node receives {@code 1 / 3}, the inverse of its graph's node count. Max pool: each graph's gradient goes to
         * the first of its nodes holding the maximum.
         */
        static void exact(ComputeBackend be, String what) {
            float[] x = {1, 2, 10, 20, 100, 200, 1000, 2000};
            GCNConv gcn = new GCNConv(2, 2, false, false);
            gcn.initialize(be, new Random(18));
            try (GraphData g = Scenarios.graph(be, x, 4, new int[]{-3, 9, 1, 2, 0}, new int[]{1, 1, -1, 4, 2},
                null)) {
                Scenarios.store(gcn.weight(), new float[]{1, 0, 0, 1});
                float[] gx = inputGradient(what + " GCNConv", g, leaf -> gcn.forward(g, leaf));
                TrainParity.assertBitIdentical(what + " GCNConv input gradient", new float[]{2, 2, 0, 0, 0, 0, 1, 1},
                    gx);
                TrainParity.assertBitIdentical(what + " GCNConv weight gradient",
                    new float[]{1002, 1002, 2004, 2004}, finiteGradient(what + " GCNConv weight", gcn.weight()));
            } finally {
                gcn.close();
            }

            float[] features = features();
            float[] batch = {0, 2, 0, 2, 2, 0};
            try (GraphData g = Scenarios.graph(be, features, N, SRC, DST, batch)) {
                float[] mean = new float[N * F];
                Arrays.fill(mean, 1.0f / 3.0f);
                TrainParity.assertDivSqrt(what + " GlobalMeanPool input gradient", mean,
                    inputGradient(what + " GlobalMeanPool", g, leaf -> new GlobalMeanPool().forward(g, leaf)));
                TrainParity.assertBitIdentical(what + " GlobalMaxPool input gradient", maxRouting(features, batch, 3),
                    inputGradient(what + " GlobalMaxPool", g, leaf -> new GlobalMaxPool().forward(g, leaf)));
            }
        }

        /**
         * GAT records exactly two segment reductions, the softmax denominator and the aggregation: the per-target
         * maximum is taken from detached scores. Releasing the output leaves the graph at its previous size.
         */
        static void gatShiftDetached(ComputeBackend be, String what) {
            GATConv m = new GATConv(F, 3, 2, true, true, 0.2f, 0.0f);
            m.initialize(be, new Random(36));
            ComputationGraph recorder = ComputationGraph.current();
            try (GraphData g = Scenarios.graph(be, features(), N, SRC, DST, null)) {
                int before = recorder.size();
                Tensor out = m.forward(g);
                try {
                    List<ComputationGraph.GraphNode> nodes = recorder.getNodes();
                    int segments = 0;
                    for (ComputationGraph.GraphNode node : nodes.subList(before, nodes.size())) {
                        if (node.gradFn() instanceof IndexGradFunctions.SegmentReduceGrad) {
                            segments++;
                        }
                    }
                    assertEquals(what + " GATConv recorded segment reductions", 2, segments);
                    recorder.release(out);
                    assertEquals(what + " GATConv graph size after release", before, recorder.size());
                } finally {
                    out.close();
                }
            } finally {
                m.close();
            }
        }

        /**
         * The ratio rule's gradients under each gate, on the unbatched graph of the scenario features (3 of 6 nodes
         * kept). For {@code L = sum(out)}, {@code out[i] = x[n_i] * act(s_i)} with {@code s_i = x[n_i] . p / ||p||}
         * and {@code S_i = sum_f x[n_i, f]}: {@code dL/dp = sum_i S_i act'(s_i) (x[n_i] / ||p|| - (x[n_i] . p) p /
         * ||p||^3)}, which differentiates through the norm of the projection; {@code dL/dx[n_i, f] = act(s_i) + S_i
         * act'(s_i) p[f] / ||p||}, and 0 for the nodes that are not kept.
         */
        static void topKGate(ComputeBackend be, String what) {
            float[] x = features();
            float[] w = TOP_K_WEIGHT;
            for (TopKPool.Nonlinearity act : TopKPool.Nonlinearity.values()) {
                String label = what + " TopKPool " + act;
                TopKPool m = new TopKPool(F, 0.5f, act);
                m.initialize(be, new Random(38));
                Scenarios.store(m.weight(), w);
                double[][] ref = topKGateReference(x, w, act);
                try (GraphData g = Scenarios.graph(be, x, N, SRC, DST, null)) {
                    backwardOfSum(label, m.forward(g).pooledFeatures());
                    Scenarios.assertClose(label + " weight gradient", ref[0],
                        finiteGradient(label + " weight", m.weight()));
                    m.zeroGrad();
                    Scenarios.assertClose(label + " input gradient", ref[1],
                        inputGradient(label, g, leaf -> m.forward(g, leaf).pooledFeatures()));
                } finally {
                    m.close();
                }
            }
        }

        /** {weight gradient, node feature gradient} of {@link #topKGate}, in double precision. */
        private static double[][] topKGateReference(float[] x, float[] w, TopKPool.Nonlinearity act) {
            double[][] forward = Reference.topK(x, w, null, 1, 0.5f, act);
            double norm = 0;
            for (float v : w) {
                norm += (double) v * v;
            }
            norm = Math.sqrt(norm);
            double[] gw = new double[F];
            double[] gx = new double[N * F];
            for (double id : forward[1]) {
                int n = (int) id;
                double dot = 0;
                double rowSum = 0;
                for (int f = 0; f < F; f++) {
                    dot += (double) x[n * F + f] * w[f];
                    rowSum += x[n * F + f];
                }
                double gate = forward[2][n];
                double slope = act == TopKPool.Nonlinearity.TANH ? 1 - gate * gate : gate * (1 - gate);
                for (int f = 0; f < F; f++) {
                    gw[f] += rowSum * slope * (x[n * F + f] / norm - dot * w[f] / (norm * norm * norm));
                    gx[n * F + f] = gate + rowSum * slope * w[f] / norm;
                }
            }
            return new double[][]{gw, gx};
        }

        /**
         * Without edges the zero aggregate is a constant: GAT's transform records nothing, so only the bias receives
         * a gradient and the backward leaves the graph at its previous size.
         */
        static void edgelessRecording(ComputeBackend be, String what) {
            GATConv m = new GATConv(F, 3, 1, true);
            m.initialize(be, new Random(37));
            ComputationGraph recorder = ComputationGraph.current();
            try (GraphData g = Scenarios.graph(be, new int[0], new int[0])) {
                int before = recorder.size();
                backwardOfSum(what + " edgeless GATConv", m.forward(g));
                assertEquals(what + " edgeless GATConv graph size after backward", before, recorder.size());
                assertNull(what + " edgeless GATConv weight gradient", m.weight().grad());
                float[] ones = new float[3];
                Arrays.fill(ones, (float) N);
                TrainParity.assertBitIdentical(what + " edgeless GATConv bias gradient", ones,
                    finiteGradient(what + " edgeless GATConv bias", m.bias()));
            } finally {
                m.close();
            }
        }

        /** Gradient comparison: {@code |dev - cpu| <= 1e-4 * max(1, max |cpu|)} element by element. */
        static void assertGradientClose(String what, float[] cpu, float[] dev) {
            assertEquals(what + ": length", cpu.length, dev.length);
            double scale = 1.0;
            for (float v : cpu) {
                scale = Math.max(scale, Math.abs(v));
            }
            double bound = TOL * scale;
            for (int i = 0; i < cpu.length; i++) {
                if (!(Math.abs(dev[i] - cpu[i]) <= bound)) {
                    fail(what + ": element " + i + ": device " + dev[i] + ", cpu path " + cpu[i]);
                }
            }
        }

        private static float[] features() {
            return TrainParity.uniform(FEATURE_SEED, N * F, -1f, 1f);
        }

        /** Checks that {@code out} requires grad, then back-propagates {@code sum(out)} and closes both. */
        private static void backwardOfSum(String what, Tensor out) {
            try {
                assertTrue(what + ": output requires grad", out.requiresGrad());
                try (Tensor loss = TensorOps.sum(out)) {
                    loss.backward();
                }
            } finally {
                out.close();
            }
        }

        /** The gradient of {@code sum(forward(x))} with respect to the graph's node features, as a leaf. */
        private static float[] inputGradient(String what, GraphData g, Function<Tensor, Tensor> forward) {
            ComputationGraph recorder = ComputationGraph.current();
            int before = recorder.size();
            Tensor leaf = g.nodeFeatures().withRequiresGrad(true);
            try {
                backwardOfSum(what, forward.apply(leaf));
                assertEquals(what + ": graph size after backward", before, recorder.size());
                Tensor grad = leaf.grad();
                assertNotNull(what + ": input gradient", grad);
                try {
                    return grad.toFloatArray();
                } finally {
                    leaf.setGrad(null);
                    grad.close();
                }
            } finally {
                leaf.close();
            }
        }

        /** The parameter's gradient on the host, checked to exist, to have its shape and to be finite. */
        private static float[] finiteGradient(String what, Parameter p) {
            Tensor grad = p.grad();
            assertNotNull(what + ": gradient", grad);
            assertEquals(what + ": gradient shape", p.tensor().shape(), grad.shape());
            float[] values = grad.toFloatArray();
            for (int i = 0; i < values.length; i++) {
                assertTrue(what + ": gradient element " + i + " is " + values[i], Float.isFinite(values[i]));
            }
            return values;
        }

        /** One at the first node (ascending id) of each graph holding the graph's maximum, per feature. */
        private static float[] maxRouting(float[] x, float[] batch, int graphs) {
            float[] routed = new float[x.length];
            for (int gr = 0; gr < graphs; gr++) {
                for (int f = 0; f < F; f++) {
                    int best = -1;
                    for (int n = 0; n < N; n++) {
                        if ((int) batch[n] == gr && (best < 0 || x[n * F + f] > x[best * F + f])) {
                            best = n;
                        }
                    }
                    if (best >= 0) {
                        routed[best * F + f] = 1.0f;
                    }
                }
            }
            return routed;
        }
    }

    // ------------------------------------------------------------------ module settings

    /**
     * The settings GATConv and TopKPool apply: GAT's attention dropout (training mode only, a replayed host mask on
     * the normalized coefficients) and TopKPool's min-score rule (a per-graph softmax and threshold).
     */
    static final class Settings {
        private Settings() {
        }

        /**
         * In evaluation mode a layer with attention dropout is bit-identical to one without (same seed, so same
         * parameters: the dropout seed is drawn after them); in training mode its output is the GAT formula with each
         * coefficient multiplied by the module's mask, whose entries are 0 or {@code 1 / (1 - 0.5) = 2}.
         */
        static void gatAttentionDropout(ComputeBackend be, String what) {
            int heads = 2;
            int out = 3;
            GATConv dropped = new GATConv(F, out, heads, true, true, 0.2f, 0.5f);
            GATConv plain = new GATConv(F, out, heads, true, true, 0.2f, 0.0f);
            dropped.initialize(be, new Random(12));
            plain.initialize(be, new Random(12));
            try (GraphData g = Scenarios.graph(be, SRC, DST)) {
                float[] bias = TrainParity.uniform(13, heads * out, -1f, 1f);
                Scenarios.store(dropped.bias(), bias);
                Scenarios.store(plain.bias(), bias);
                dropped.eval();
                TrainParity.assertBitIdentical(what + " GATConv eval mode ignores the dropout",
                    Scenarios.run(plain.forward(g)), Scenarios.run(dropped.forward(g)));

                dropped.train();
                float[] trained = Scenarios.run(dropped.forward(g));
                Dropout attention = (Dropout) dropped.namedChildren().get(GATConv.ATTENTION_DROPOUT);
                assertNotNull(what + " attention dropout is a registered child", attention);
                float[] mask = attention.getLastMask().toFloatArray();
                assertEquals(what + " one mask entry per edge and head", SRC.length * heads, mask.length);
                boolean kept = false;
                boolean zeroed = false;
                for (float v : mask) {
                    assertTrue(what + " mask entry " + v, v == 0.0f || v == 2.0f);
                    kept |= v == 2.0f;
                    zeroed |= v == 0.0f;
                }
                assertTrue(what + " the mask keeps and drops coefficients", kept && zeroed);
                double[] ref = gatMasked(g.nodeFeatures().toFloatArray(), dropped.weight().tensor().toFloatArray(),
                    dropped.attSrc().tensor().toFloatArray(), dropped.attDst().tensor().toFloatArray(), heads, out,
                    0.2f, bias, mask);
                Scenarios.assertClose(what + " GATConv attention dropout", ref, trained);
            } finally {
                dropped.close();
                plain.close();
            }
        }

        /**
         * In grad mode a training forward records exactly one {@code MaskScaleGrad} (the attention dropout) and an
         * evaluation forward none; releasing either output leaves the graph at its previous size.
         */
        static void gatDropoutRecording(ComputeBackend be, String what) {
            GATConv m = new GATConv(F, 3, 2, true, true, 0.2f, 0.5f);
            m.initialize(be, new Random(38));
            ComputationGraph recorder = ComputationGraph.current();
            try (GraphData g = Scenarios.graph(be, Gradients.features(), N, SRC, DST, null)) {
                for (boolean training : new boolean[]{true, false}) {
                    m.train(training);
                    int before = recorder.size();
                    Tensor out = m.forward(g);
                    try {
                        List<ComputationGraph.GraphNode> nodes = recorder.getNodes();
                        int masks = 0;
                        for (ComputationGraph.GraphNode node : nodes.subList(before, nodes.size())) {
                            if (node.gradFn() instanceof IndexGradFunctions.MaskScaleGrad) {
                                masks++;
                            }
                        }
                        assertEquals(what + " GATConv training=" + training + " recorded dropouts", training ? 1 : 0,
                            masks);
                        recorder.release(out);
                        assertEquals(what + " GATConv graph size after release", before, recorder.size());
                    } finally {
                        out.close();
                    }
                }
            } finally {
                m.close();
            }
        }

        /**
         * The min-score rule on a batch of three graphs (graph 1 empty, node 4 in none) with the scores of
         * {@link Scenarios#pools}: graph 0 normalizes to {0.331, 0.551, 0.117} at nodes {0, 2, 5} and graph 2 to
         * {0.338, 0.662} at nodes {1, 3}, so {@code minScore = 0.3} keeps nodes 0 to 3 and {@code +inf} only each
         * graph's best node (2 and 3). The kept rows are gated by their normalized score; node 4's score is NaN.
         */
        static void topKMinScore(ComputeBackend be, String what) {
            float[] x = TrainParity.uniform(16, N * F, -1f, 1f);
            float[] batch = {0, 2, 0, 2, -1, 0};
            float[] w = TOP_K_WEIGHT;
            float[] minScores = {0.3f, Float.POSITIVE_INFINITY};
            int[][] expected = {{0, 1, 2, 3}, {2, 3}};
            for (int c = 0; c < minScores.length; c++) {
                String label = what + " TopKPool minScore=" + minScores[c];
                TopKPool m = new TopKPool(F, 0.5f, minScores[c]);
                m.initialize(be, new Random(17));
                try (GraphData g = Scenarios.graph(be, x, N, SRC, DST, batch)) {
                    Scenarios.store(m.weight(), w);
                    assertTrue(label + " uses the min-score rule", m.hasMinScore());
                    TopKPool.TopKPoolResult r = m.forward(g);
                    double[][] ref = minScoreReference(x, w, batch, 3, minScores[c]);
                    assertArrayEquals(label + " selection", expected[c], r.selectedIndices());
                    assertEquals(label + " reference selection", expected[c].length, ref[1].length);
                    Scenarios.assertClose(label + " features", ref[0], Scenarios.run(r.pooledFeatures()));
                    float[] scores = r.scores();
                    assertTrue(label + " node outside every graph has a NaN score", Float.isNaN(scores[4]));
                    for (int n = 0; n < N; n++) {
                        if (n != 4) {
                            Scenarios.assertClose(label + " normalized score " + n, new double[]{ref[2][n]},
                                new float[]{scores[n]});
                        }
                    }
                } finally {
                    m.close();
                }
            }
        }

        /**
         * A NaN min score, a null TopKPool nonlinearity and a dropout outside {@code [0, 1)} are rejected at
         * construction; TopKPool gates with tanh unless told otherwise.
         */
        static void rejectedSettings() {
            for (float dropout : new float[]{-0.1f, 1.0f, Float.NaN}) {
                try {
                    new GATConv(F, 3, 1, true, true, 0.2f, dropout);
                    fail("GATConv dropout " + dropout + ": expected IllegalArgumentException");
                } catch (IllegalArgumentException expected) {
                    assertTrue(expected.getMessage(), expected.getMessage().contains("dropout"));
                }
            }
            try {
                new TopKPool(F, 0.5f, Float.NaN);
                fail("TopKPool minScore NaN: expected IllegalArgumentException");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("minScore"));
            }
            assertFalse("the two-argument TopKPool uses the ratio rule", new TopKPool(F, 0.5f).hasMinScore());
            assertEquals("TopKPool gates with tanh by default", TopKPool.Nonlinearity.TANH,
                new TopKPool(F, 0.5f).nonlinearity());
            assertEquals("the min-score TopKPool keeps the tanh default", TopKPool.Nonlinearity.TANH,
                new TopKPool(F, 0.5f, 0.2f).nonlinearity());
            try {
                new TopKPool(F, 0.5f, (TopKPool.Nonlinearity) null);
                fail("TopKPool null nonlinearity: expected NullPointerException");
            } catch (NullPointerException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("nonlinearity"));
            }
        }

        /**
         * GAT with concatenated heads over the edges {@code SRC -> DST}, each coefficient {@code alpha[e, k]}
         * multiplied by {@code mask[e * heads + k]}.
         */
        private static double[] gatMasked(float[] x, float[] w, float[] as, float[] ad, int heads, int out,
                                          float slope, float[] b, float[] mask) {
            int ho = heads * out;
            double[] h = Reference.matmul(Reference.toDouble(x), N, F, w, ho);
            int edges = SRC.length;
            double[] score = new double[edges * heads];
            double[] max = new double[N * heads];
            Arrays.fill(max, Double.NEGATIVE_INFINITY);
            for (int e = 0; e < edges; e++) {
                for (int k = 0; k < heads; k++) {
                    double logit = 0;
                    for (int f = 0; f < out; f++) {
                        int c = k * out + f;
                        logit += h[SRC[e] * ho + c] * as[c] + h[DST[e] * ho + c] * ad[c];
                    }
                    double s = logit > 0 ? logit : slope * logit;
                    score[e * heads + k] = s;
                    max[DST[e] * heads + k] = Math.max(max[DST[e] * heads + k], s);
                }
            }
            double[] sum = new double[N * heads];
            for (int e = 0; e < edges; e++) {
                for (int k = 0; k < heads; k++) {
                    sum[DST[e] * heads + k] += Math.exp(score[e * heads + k] - max[DST[e] * heads + k]);
                }
            }
            double[] agg = new double[N * ho];
            for (int e = 0; e < edges; e++) {
                for (int k = 0; k < heads; k++) {
                    int t = DST[e] * heads + k;
                    double alpha = Math.exp(score[e * heads + k] - max[t]) / sum[t] * mask[e * heads + k];
                    for (int f = 0; f < out; f++) {
                        agg[DST[e] * ho + k * out + f] += alpha * h[SRC[e] * ho + k * out + f];
                    }
                }
            }
            return Reference.addBias(agg, b);
        }

        /**
         * {pooled features, kept node ids, normalized scores} of the min-score rule: the softmax of the raw projections
         * {@code x[n] . w} over each graph's nodes (PyG's {@code softmax(score, batch)}, no nonlinearity), a node kept
         * when its normalized score exceeds {@code min(minScore, graphMax - 1e-7)}, in ascending id, its row scaled by
         * that score. Nodes outside every graph get a NaN score.
         */
        private static double[][] minScoreReference(float[] x, float[] w, float[] batch, int graphs,
                                                    float minScore) {
            double[] score = new double[N];
            for (int n = 0; n < N; n++) {
                score[n] = 0;
                for (int f = 0; f < F; f++) {
                    score[n] += (double) x[n * F + f] * w[f];
                }
            }
            double[] max = new double[graphs];
            double[] sum = new double[graphs];
            Arrays.fill(max, Double.NEGATIVE_INFINITY);
            for (int n = 0; n < N; n++) {
                int g = (int) batch[n];
                if (g >= 0) {
                    max[g] = Math.max(max[g], score[n]);
                }
            }
            for (int n = 0; n < N; n++) {
                int g = (int) batch[n];
                if (g >= 0) {
                    sum[g] += Math.exp(score[n] - max[g]);
                }
            }
            double[] normalized = new double[N];
            double[] best = new double[graphs];
            Arrays.fill(best, Double.NEGATIVE_INFINITY);
            for (int n = 0; n < N; n++) {
                int g = (int) batch[n];
                normalized[n] = g >= 0 ? Math.exp(score[n] - max[g]) / sum[g] : Double.NaN;
                if (g >= 0) {
                    best[g] = Math.max(best[g], normalized[n]);
                }
            }
            List<Integer> kept = new ArrayList<>();
            for (int n = 0; n < N; n++) {
                int g = (int) batch[n];
                if (g >= 0 && normalized[n] > Math.min(minScore, best[g] - 1e-7)) {
                    kept.add(n);
                }
            }
            double[] pooled = new double[kept.size() * F];
            double[] ids = new double[kept.size()];
            for (int i = 0; i < kept.size(); i++) {
                int n = kept.get(i);
                ids[i] = n;
                for (int f = 0; f < F; f++) {
                    pooled[i * F + f] = x[n * F + f] * normalized[n];
                }
            }
            return new double[][]{pooled, ids, normalized};
        }
    }

    // ------------------------------------------------------------------ double-precision references

    /** The module formulas in double precision, on row-major arrays. */
    static final class Reference {
        private Reference() {
        }

        static double[] gcn(float[] x, int[] src, int[] dst, float[] w, int out, float[] b, boolean normalize) {
            int[] deg = new int[N];
            for (int e = 0; e < src.length; e++) {
                deg[src[e]]++;
                deg[dst[e]]++;
            }
            double[] agg = new double[N * F];
            for (int e = 0; e < src.length; e++) {
                double c = normalize ? 1.0 / Math.sqrt(deg[src[e]]) / Math.sqrt(deg[dst[e]]) : 1.0;
                for (int f = 0; f < F; f++) {
                    agg[dst[e] * F + f] += c * x[src[e] * F + f];
                }
            }
            return addBias(matmul(agg, N, F, w, out), b);
        }

        static double[] gat(float[] x, int[] src, int[] dst, float[] w, float[] as, float[] ad, int heads, int out,
                            float slope, boolean concat, float[] b) {
            int ho = heads * out;
            double[] h = matmul(toDouble(x), N, F, w, ho);
            int edges = src.length;
            double[] score = new double[edges * heads];
            double[] max = new double[N * heads];
            Arrays.fill(max, Double.NEGATIVE_INFINITY);
            for (int e = 0; e < edges; e++) {
                for (int k = 0; k < heads; k++) {
                    double logit = 0;
                    for (int f = 0; f < out; f++) {
                        int c = k * out + f;
                        logit += h[src[e] * ho + c] * as[c] + h[dst[e] * ho + c] * ad[c];
                    }
                    double s = logit > 0 ? logit : slope * logit;
                    score[e * heads + k] = s;
                    max[dst[e] * heads + k] = Math.max(max[dst[e] * heads + k], s);
                }
            }
            double[] sum = new double[N * heads];
            for (int e = 0; e < edges; e++) {
                for (int k = 0; k < heads; k++) {
                    sum[dst[e] * heads + k] += Math.exp(score[e * heads + k] - max[dst[e] * heads + k]);
                }
            }
            double[] agg = new double[N * ho];
            for (int e = 0; e < edges; e++) {
                for (int k = 0; k < heads; k++) {
                    int t = dst[e] * heads + k;
                    double alpha = Math.exp(score[e * heads + k] - max[t]) / sum[t];
                    for (int f = 0; f < out; f++) {
                        agg[dst[e] * ho + k * out + f] += alpha * h[src[e] * ho + k * out + f];
                    }
                }
            }
            if (concat) {
                return addBias(agg, b);
            }
            double[] mean = new double[N * out];
            for (int n = 0; n < N; n++) {
                for (int f = 0; f < out; f++) {
                    for (int k = 0; k < heads; k++) {
                        mean[n * out + f] += agg[n * ho + k * out + f] / heads;
                    }
                }
            }
            return addBias(mean, b);
        }

        static double[] sage(float[] x, int[] src, int[] dst, float[] ws, float[] wn, int out, float[] b, boolean max,
                             boolean normalize) {
            double[] agg = new double[N * F];
            int[] count = new int[N];
            if (max) {
                Arrays.fill(agg, Double.NEGATIVE_INFINITY);
            }
            for (int e = 0; e < src.length; e++) {
                count[dst[e]]++;
                for (int f = 0; f < F; f++) {
                    int i = dst[e] * F + f;
                    agg[i] = max ? Math.max(agg[i], x[src[e] * F + f]) : agg[i] + x[src[e] * F + f];
                }
            }
            for (int n = 0; n < N; n++) {
                for (int f = 0; f < F; f++) {
                    int i = n * F + f;
                    agg[i] = count[n] == 0 ? 0 : max ? agg[i] : agg[i] / count[n];
                }
            }
            double[] self = matmul(toDouble(x), N, F, ws, out);
            double[] neigh = matmul(agg, N, F, wn, out);
            for (int i = 0; i < self.length; i++) {
                self[i] += neigh[i];
            }
            double[] r = addBias(self, b);
            if (normalize) {
                for (int n = 0; n < N; n++) {
                    double sq = 0;
                    for (int f = 0; f < out; f++) {
                        sq += r[n * out + f] * r[n * out + f];
                    }
                    double norm = Math.sqrt(sq) + 1e-8;
                    for (int f = 0; f < out; f++) {
                        r[n * out + f] /= norm;
                    }
                }
            }
            return r;
        }

        static double[] gin(float[] x, int[] src, int[] dst, float eps, float[] w1, float[] b1, float[] w2,
                            float[] b2, int out) {
            double[] z = new double[N * F];
            for (int i = 0; i < z.length; i++) {
                z[i] = (1.0 + eps) * x[i];
            }
            for (int e = 0; e < src.length; e++) {
                for (int f = 0; f < F; f++) {
                    z[dst[e] * F + f] += x[src[e] * F + f];
                }
            }
            int hidden = b1.length;
            double[] h = addBias(matmul(z, N, F, w1, hidden), b1);
            for (int i = 0; i < h.length; i++) {
                h[i] = Math.max(0, h[i]);
            }
            return addBias(matmul(h, N, hidden, w2, out), b2);
        }

        /**
         * {pooled features, selected node ids, every node's score} of the ratio rule (PyG {@code SelectTopK}): the
         * score {@code act(x[n] . w / ||w||)}; per graph of {@code batch} (one graph when null), its top
         * {@code ceil(ratio * n_g)} nodes by descending score (stable over ascending ids), graph by graph in ascending
         * order, each row multiplied by its score.
         */
        static double[][] topK(float[] x, float[] w, float[] batch, int graphs, float ratio,
                               TopKPool.Nonlinearity act) {
            double norm = 0;
            for (float v : w) {
                norm += (double) v * v;
            }
            norm = Math.sqrt(norm);
            double[] score = new double[N];
            Integer[] order = new Integer[N];
            int[] graphOf = new int[N];
            int[] size = new int[graphs];
            for (int n = 0; n < N; n++) {
                double dot = 0;
                for (int f = 0; f < F; f++) {
                    dot += (double) x[n * F + f] * w[f];
                }
                score[n] = act == TopKPool.Nonlinearity.TANH ? Math.tanh(dot / norm)
                    : 1.0 / (1.0 + Math.exp(-dot / norm));
                order[n] = n;
                graphOf[n] = batch == null ? 0 : (int) batch[n];
                size[graphOf[n]]++;
            }
            Arrays.sort(order, (a, c) -> graphOf[a] != graphOf[c] ? Integer.compare(graphOf[a], graphOf[c])
                : Double.compare(score[c], score[a]));
            List<Integer> kept = new ArrayList<>();
            int start = 0;
            for (int g = 0; g < graphs; g++) {
                int k = (int) Math.ceil(size[g] * (double) ratio);
                for (int i = 0; i < k; i++) {
                    kept.add(order[start + i]);
                }
                start += size[g];
            }
            int k = kept.size();
            double[] pooled = new double[k * F];
            double[] ids = new double[k];
            for (int i = 0; i < k; i++) {
                int n = kept.get(i);
                ids[i] = n;
                for (int f = 0; f < F; f++) {
                    pooled[i * F + f] = x[n * F + f] * score[n];
                }
            }
            return new double[][]{pooled, ids, score};
        }

        static double[] pool(float[] x, float[] batch, int graphs, boolean max) {
            double[] r = new double[graphs * F];
            int[] count = new int[graphs];
            if (max) {
                Arrays.fill(r, Double.NEGATIVE_INFINITY);
            }
            for (int n = 0; n < N; n++) {
                int g = batch == null ? 0 : (int) batch[n];
                count[g]++;
                for (int f = 0; f < F; f++) {
                    r[g * F + f] = max ? Math.max(r[g * F + f], x[n * F + f]) : r[g * F + f] + x[n * F + f];
                }
            }
            for (int g = 0; g < graphs; g++) {
                for (int f = 0; f < F; f++) {
                    r[g * F + f] = count[g] == 0 ? 0 : max ? r[g * F + f] : r[g * F + f] / count[g];
                }
            }
            return r;
        }

        private static double[] matmul(double[] a, int m, int k, float[] w, int n) {
            double[] r = new double[m * n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double s = 0;
                    for (int p = 0; p < k; p++) {
                        s += a[i * k + p] * w[p * n + j];
                    }
                    r[i * n + j] = s;
                }
            }
            return r;
        }

        private static double[] addBias(double[] r, float[] b) {
            for (int i = 0; i < r.length; i++) {
                r[i] += b[i % b.length];
            }
            return r;
        }

        private static double[] toDouble(float[] x) {
            double[] r = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                r[i] = x[i];
            }
            return r;
        }
    }
}
