package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of the graph modules: for every module class with parameters,
 * the gradients of {@code sum(output * probe)} with respect to the node features ({@link GradientCheckUtil#gradCheck})
 * and to all of the module's parameters ({@link GradientCheckUtil#gradCheckParameters}) match central differences with
 * step 1e-3 within 1e-2, on the host reference backend and on every available GPU API ({@link TrainParity}); the
 * parameter-free readouts are checked on their input. The probe is a fixed tensor in [-1, 1].
 *
 * <p><b>Graph.</b> Five nodes and ten directed edges; every node has one to three incoming edges, so every
 * aggregation (sum, mean, max, edge softmax) runs over non-empty segments. The edge index is a constant. The node
 * features are {@code lo + (hi - lo) r / 14} over a permutation {@code r} of 0..14: all fifteen values are distinct and
 * at least {@code (hi - lo) / 14} apart. The modules are built from a seeded {@link Random}, every bias gets seeded
 * values, attention dropout is 0 and the modules run in evaluation mode, so every evaluation computes the same
 * function.</p>
 *
 * <p><b>Kinks.</b> The kink-free paths (GCN normalization, mean aggregation, L2 normalization, the TopK gate) need
 * nothing. Max aggregation and the max readout select among raw feature values at least 0.14 apart, which a
 * perturbation of 1e-3 cannot reorder. GAT runs on features in [0.2, 1] with positive weights and head-0 attention
 * vectors in [0.2, 0.6] and head-1 vectors in [-0.6, -0.2], so every head-0 logit is at least 0.048 and every head-1
 * logit at most -0.048, while a perturbation moves a logit by at most 3e-3: both LeakyReLU branches are checked and
 * neither is crossed. GIN runs on features in [0.2, 1] with a first MLP layer whose columns alternate between entries
 * in [0.1, 0.5] and in [-0.5, -0.1] and biases below 0.02, so every hidden pre-activation is at least 0.1 away from the
 * ReLU kink (the aggregate of a node is at least 0.4 per feature). TopKPool ranks nodes by {@code x . p / ||p||}
 * (its tanh gate is monotone and smooth), which is dominated by a first feature spaced 0.4 apart across nodes with
 * weight 1 and the other weights 0.05 and -0.05, so the projections are at least 0.19 apart and neither the selection
 * nor its order changes.</p>
 *
 * <p><b>Lifetime.</b> Input checks run with the module frozen, so no parameter slot is written; parameter checks
 * restore every slot. Each check leaves the thread's graph at the size it had before the check.</p>
 */
@RunWith(Parameterized.class)
public class GraphGradCheckTest {

    private static final long SEED = 6262L;
    private static final float EPSILON = 1e-3f;
    private static final float TOLERANCE = 1e-2f;
    private static final int MAX_ELEMENTS = 6;
    private static final int NODES = 5;
    private static final int FEATURES = 3;
    private static final int[] SOURCES = {0, 2, 3, 1, 4, 0, 1, 2, 4, 3};
    private static final int[] TARGETS = {1, 1, 1, 0, 0, 2, 3, 3, 3, 4};

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> backends() {
        List<Object[]> rows = new ArrayList<>();
        rows.add(new Object[]{"host", null});
        for (TrainParity.Api api : TrainParity.Api.values()) {
            rows.add(new Object[]{api.name(), api});
        }
        return rows;
    }

    private final String name;
    private final TrainParity.Api api;
    private final List<AutoCloseable> owned = new ArrayList<>();
    private TrainParity.Session session;
    private ComputeBackend backend;

    public GraphGradCheckTest(String name, TrainParity.Api api) {
        this.name = name;
        this.api = api;
    }

    @Before
    public void setUp() {
        ComputationGraph.setGradEnabled(true);
        if (api == null) {
            backend = TrainParity.hostBackend();
        } else {
            session = TrainParity.open(api);
            backend = session.backend();
        }
    }

    @After
    public void tearDown() throws Exception {
        Exception failure = null;
        for (int i = owned.size() - 1; i >= 0; i--) {
            try {
                owned.get(i).close();
            } catch (Exception e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        owned.clear();
        if (session != null) {
            session.close();
        } else if (backend != null) {
            TrainGpu.release(backend);
        }
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ message passing layers

    /** With symmetric normalization, then plain summation. */
    @Test
    public void gcnConv() {
        float[] x = distinct(-1f, 1f);
        GCNConv normalized = keep(new GCNConv(FEATURES, 4));
        prepare(normalized, (b, r) -> normalized.initialize(b, r));
        checkLayer("normalized GCNConv", normalized, x, 4, 11);

        GCNConv summed = keep(new GCNConv(FEATURES, 4, true, false));
        prepare(summed, (b, r) -> summed.initialize(b, r));
        checkLayer("summing GCNConv", summed, x, 4, 13);
    }

    /** Two heads, concatenated and then averaged, with one head in each LeakyReLU branch (class javadoc). */
    @Test
    public void gatConv() {
        float[] x = distinct(0.2f, 1f);
        int heads = 2;
        int out = 2;
        GATConv concat = keep(new GATConv(FEATURES, out, heads, true, true, 0.2f, 0.0f));
        prepare(concat, (b, r) -> concat.initialize(b, r));
        signAttention(concat, heads, out);
        checkLayer("concatenating GATConv", concat, x, heads * out, 21);

        GATConv mean = keep(new GATConv(FEATURES, out, heads, false, true, 0.2f, 0.0f));
        prepare(mean, (b, r) -> mean.initialize(b, r));
        signAttention(mean, heads, out);
        checkLayer("averaging GATConv", mean, x, out, 23);
    }

    /** Mean aggregation with L2 normalization, then max-pool aggregation without it. */
    @Test
    public void graphSageConv() {
        float[] x = distinct(-1f, 1f);
        GraphSAGEConv mean = keep(new GraphSAGEConv(FEATURES, 4));
        prepare(mean, (b, r) -> mean.initialize(b, r));
        checkLayer("mean GraphSAGEConv", mean, x, 4, 31);

        GraphSAGEConv max = keep(new GraphSAGEConv(FEATURES, 4, GraphSAGEConv.Aggregator.MAX_POOL, true, false));
        prepare(max, (b, r) -> max.initialize(b, r));
        checkLayer("max-pool GraphSAGEConv", max, x, 4, 33);
    }

    /** A trainable {@code eps}, checked with the MLP's parameters. */
    @Test
    public void ginConv() {
        float[] x = distinct(0.2f, 1f);
        GINConv gin = keep(new GINConv(FEATURES, 4, 0.0f, true));
        prepare(gin, (b, r) -> gin.initialize(b, r));
        Parameter w1 = gin.getParameter("mlp.weight1");
        int hidden = w1.shape().dim(1);
        float[] magnitudes = TrainParity.uniform(41, FEATURES * hidden, 0.1f, 0.5f);
        float[] signed = new float[FEATURES * hidden];
        for (int i = 0; i < FEATURES; i++) {
            for (int j = 0; j < hidden; j++) {
                signed[i * hidden + j] = (j % 2 == 0 ? 1f : -1f) * magnitudes[i * hidden + j];
            }
        }
        store(w1, signed);
        store(gin.getParameter("mlp.bias1"), TrainParity.uniform(42, hidden, -0.02f, 0.02f));
        checkLayer("GINConv", gin, x, 4, 43);
    }

    // ------------------------------------------------------------------ pooling

    /** Keeps 3 of 5 nodes; the gradient reaches the features through the kept rows and through the gate scores. */
    @Test
    public void topKPool() {
        float[] x = distinct(-1f, 1f);
        for (int n = 0; n < NODES; n++) {
            x[n * FEATURES] = -0.8f + 0.4f * ((2 * n) % NODES);
        }
        TopKPool pool = keep(new TopKPool(FEATURES, 0.6f));
        prepare(pool, (b, r) -> pool.initialize(b, r));
        store(pool.weight(), new float[]{1.0f, 0.05f, -0.05f});
        GraphData graph = graph(x, null);
        Tensor features = keep(upload(x, NODES, FEATURES));
        Tensor probe = tensor(51, 3, FEATURES);
        Function<Tensor, Tensor> loss = in -> probeSum(pool.forward(graph, in).pooledFeatures(), probe);
        checkInputs("TopKPool", pool, loss, features);
        checkParameters("TopKPool", pool, () -> loss.apply(features));
    }

    /** The mean and max readouts of a batch of two graphs (nodes 0..2 and 3..4); they have no parameters. */
    @Test
    public void globalReadouts() {
        float[] x = distinct(-1f, 1f);
        GraphData graph = graph(x, new float[]{0f, 0f, 0f, 1f, 1f});
        Tensor features = keep(upload(x, NODES, FEATURES));
        Tensor probe = tensor(61, 2, FEATURES);
        GlobalMeanPool mean = keep(new GlobalMeanPool());
        checkInputs("GlobalMeanPool", mean, in -> probeSum(mean.forward(graph, in), probe), features);
        GlobalMaxPool max = keep(new GlobalMaxPool());
        checkInputs("GlobalMaxPool", max, in -> probeSum(max.forward(graph, in), probe), features);
    }

    // ------------------------------------------------------------------ shared cases

    /** A message-passing layer on the fixed graph: output {@code [NODES, outFeatures]}. */
    private void checkLayer(String what, MessagePassing layer, float[] x, int outFeatures, long probeSeed) {
        GraphData graph = graph(x, null);
        Tensor features = keep(upload(x, NODES, FEATURES));
        Tensor probe = tensor(probeSeed, NODES, outFeatures);
        Function<Tensor, Tensor> loss = in -> probeSum(layer.forward(graph, in), probe);
        checkInputs(what, layer, loss, features);
        checkParameters(what, layer, () -> loss.apply(features));
    }

    /** Positive weights, head-0 attention vectors in [0.2, 0.6] and head-1 vectors in [-0.6, -0.2]. */
    private void signAttention(GATConv gat, int heads, int out) {
        Parameter weight = gat.weight();
        store(weight, TrainParity.uniform(71, (int) weight.numElements(), 0.1f, 0.5f));
        Parameter[] attention = {gat.attSrc(), gat.attDst()};
        for (int a = 0; a < attention.length; a++) {
            float[] values = TrainParity.uniform(72 + a, heads * out, 0.2f, 0.6f);
            for (int h = 1; h < heads; h += 2) {
                for (int f = 0; f < out; f++) {
                    values[h * out + f] = -values[h * out + f];
                }
            }
            store(attention[a], values);
        }
    }

    // ------------------------------------------------------------------ checks

    /** The node-feature gradients of {@code loss}, with {@code module}'s parameters frozen meanwhile. */
    private void checkInputs(String what, Module module, Function<Tensor, Tensor> loss, Tensor features) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result;
        module.setRequiresGrad(false);
        try {
            result = GradientCheckUtil.gradCheck(in -> loss.apply(in.get(0)), Collections.singletonList(features),
                EPSILON, TOLERANCE);
        } finally {
            module.setRequiresGrad(true);
        }
        assertTrue(name + " " + what + " input gradients: " + result, result.passed());
        assertEquals(name + " " + what + " input check leaves the graph as it was", before, graph.size());
    }

    /** The gradients of every parameter of {@code module} for {@code loss}. */
    private void checkParameters(String what, Module module, Supplier<Tensor> loss) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(loss, module,
            module.parameters(), EPSILON, TOLERANCE, MAX_ELEMENTS);
        assertTrue(name + " " + what + " parameter gradients: " + result, result.passed());
        assertEquals(name + " " + what + " parameter check leaves the graph as it was", before, graph.size());
    }

    /** {@code sum(output * probe)} as a new scalar; {@code output} is closed. */
    static Tensor probeSum(Tensor output, Tensor probe) {
        try (Tensor out = output;
             Tensor weighted = TensorOps.mul(out, probe)) {
            return TensorOps.sum(weighted);
        }
    }

    // ------------------------------------------------------------------ setup

    /**
     * Initializes a module on the backend from the seed, gives every parameter whose name ends in {@code bias} (with
     * an optional index) seeded values in [-0.5, 0.5] and puts it in eval mode.
     */
    private void prepare(Module module, Initializer init) {
        init.initialize(backend, new Random(SEED));
        for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
            String key = e.getKey();
            String leaf = key.substring(key.lastIndexOf('.') + 1);
            if (leaf.matches("bias\\d*")) {
                Parameter p = e.getValue();
                store(p, TrainParity.uniform(SEED + key.hashCode(), (int) p.numElements(), -0.5f, 0.5f));
            }
        }
        module.eval();
    }

    /**
     * {@code NODES x FEATURES} values {@code lo + (hi - lo) r / (n - 1)}, {@code r = 7 i mod n} for element {@code i}:
     * a permutation, since 7 and {@code n = 15} are coprime.
     */
    private static float[] distinct(float lo, float hi) {
        int n = NODES * FEATURES;
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = lo + (hi - lo) * ((7 * i) % n) / (n - 1);
        }
        return values;
    }

    /** The fixed graph over a copy of {@code x}, batched by {@code batch} unless it is null; closed after the test. */
    private GraphData graph(float[] x, float[] batch) {
        int[] index = new int[2 * SOURCES.length];
        System.arraycopy(SOURCES, 0, index, 0, SOURCES.length);
        System.arraycopy(TARGETS, 0, index, SOURCES.length, TARGETS.length);
        Tensor nodeFeatures = upload(x, NODES, FEATURES);
        Tensor edgeIndex = null;
        Tensor batchIndex = null;
        try {
            try (Tensor flat = Tensor.fromArray(backend, index)) {
                edgeIndex = flat.reshape(2, SOURCES.length);
            }
            batchIndex = batch == null ? null : Tensor.fromArray(backend, batch);
            return keep(new GraphData(nodeFeatures, edgeIndex, null, batchIndex));
        } catch (RuntimeException e) {
            nodeFeatures.close();
            if (edgeIndex != null) {
                edgeIndex.close();
            }
            if (batchIndex != null) {
                batchIndex.close();
            }
            throw e;
        }
    }

    /** Overwrites a parameter's values; an assignment, so it runs under {@code noGrad} (in-place guard). */
    private static void store(Parameter p, float[] values) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            GpuOptimizerOps.storeFloats(p.tensor(), values);
        }
    }

    /** A new tensor of {@code dims} uniform in [-1, 1] from {@code seed}, closed after the test. */
    private Tensor tensor(long seed, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        return keep(upload(TrainParity.uniform(seed, n, -1f, 1f), dims));
    }

    /** A new tensor of {@code dims} holding {@code data} on the backend. */
    private Tensor upload(float[] data, int... dims) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(dims);
        }
    }

    private <T extends AutoCloseable> T keep(T resource) {
        owned.add(resource);
        return resource;
    }

    /** Initializes a freshly built module on a backend. */
    private interface Initializer {
        void initialize(ComputeBackend backend, Random random);
    }
}
