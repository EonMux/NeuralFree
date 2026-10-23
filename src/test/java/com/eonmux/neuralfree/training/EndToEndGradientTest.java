package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.nn.transformer.MixtureOfExperts;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.IntFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * One training step through {@link Trainer} end to end, on the CPU reference backend and on each GPU API with the
 * train kernels: an MLP and a CNN give every parameter a non-zero gradient that
 * agrees with a separate backward pass and with central finite differences
 * ({@link GradientCheckUtil#gradCheckParameters}), and the optimizer step changes every parameter, which stays an
 * autograd leaf; with {@code accumulationSteps = 2} the applied gradient is the mean of the two batch gradients; a
 * mixture-of-experts router receives the load-balance term's gradient through {@link Module#takeAuxiliaryLoss()}; the
 * graph is empty after every batch.
 *
 * <p><b>Kinks.</b> Finite differences are exact only away from the relu kink and max-pool ties. The models and inputs
 * come from the first seed whose relu inputs all lie at least {@link #MARGIN} from 0 and whose max-pool windows have
 * their largest positive value at least {@link #MARGIN} above the next one, several times what a perturbation of
 * {@code 1e-3} moves them; the CNN's seed also keeps every feature the classifier reads alive, so every layer gets a
 * non-zero gradient, and the MLP's seed has an active unit in each hidden layer for the same reason.</p>
 */
@RunWith(Parameterized.class)
public class EndToEndGradientTest {

    private static final float EPSILON = 1e-3f;
    private static final float TOLERANCE = 1e-2f;
    private static final float MARGIN = 0.02f;
    private static final int SEED_SEARCH = 2000;
    private static final float LR = 0.1f;

    private static final int CLASSES = 3;
    private static final int MLP_IN = 6;
    private static final int MLP_HIDDEN = 8;
    private static final int MLP_BATCH = 4;
    private static final int CNN_BATCH = 2;
    private static final int CNN_SIZE = 4;
    private static final int CNN_CHANNELS = 2;
    private static final int CNN_FLAT = CNN_CHANNELS * (CNN_SIZE / 4) * (CNN_SIZE / 4);

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> targets() {
        List<Object[]> rows = new ArrayList<>();
        rows.add(new Object[]{"HOST", null});
        for (Object[] api : TrainParity.apis()) {
            rows.add(new Object[]{api[0].toString(), api[0]});
        }
        return rows;
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend backend;
    private final List<AutoCloseable> owned = new ArrayList<>();

    public EndToEndGradientTest(String name, TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
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
        try {
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
            owned.clear();
            ComputationGraph.current().clear();
        } finally {
            if (session != null) {
                session.close();
            } else if (backend != null) {
                backend.close();
            }
        }
    }

    // ------------------------------------------------------------------ MLP and CNN steps

    @Test
    public void mlpStepGivesEveryParameterItsGradientAndUpdatesIt() {
        Candidate<Mlp> mlp = kinkFree(seed -> new Mlp(backend, new Random(seed)), Shape.of(MLP_BATCH, MLP_IN));
        assertStepTrainsEveryParameter(mlp, Shape.of(MLP_IN));
    }

    @Test
    public void cnnStepGivesEveryParameterItsGradientAndUpdatesIt() {
        Candidate<Cnn> cnn = kinkFree(seed -> new Cnn(backend, new Random(seed)),
            Shape.of(CNN_BATCH, 1, CNN_SIZE, CNN_SIZE));
        assertStepTrainsEveryParameter(cnn, Shape.of(1, CNN_SIZE, CNN_SIZE));
    }

    /**
     * One {@link Trainer} batch over the candidate's whole input: the gradients the optimizer applies equal those of a
     * separate backward pass, are non-zero for every parameter and pass the finite-difference check; every parameter
     * changes by {@code -LR * gradient} and stays a leaf; no node is left after the batch.
     */
    private void assertStepTrainsEveryParameter(Candidate<? extends Module> c, Shape sampleShape) {
        Module model = c.model;
        LossFunction loss = new CrossEntropyLoss();
        int batch = c.inputShape.dim(0);
        float[] labels = labels(batch);
        Tensor x = track(tensor(c.inputs, c.inputShape));
        Tensor y = track(tensor(labels, Shape.of(batch, 1)));

        GradientCheckUtil.GradientCheckResult check = GradientCheckUtil.gradCheckParameters(
            () -> lossOf(model, loss, x, y), model, model.parameters(), EPSILON, TOLERANCE, 16);
        assertTrue(api + " finite differences: " + check, check.passed());
        assertEquals("the check leaves no node", 0, ComputationGraph.current().size());

        Map<Parameter, float[]> reference = backwardGradients(model, loss, x, y);
        Map<Parameter, float[]> before = values(model);
        TensorDataset data = track(dataset(c.inputs, labels, sampleShape, batch));
        RecordingOptimizer optimizer = new RecordingOptimizer(model.parameters(), LR);
        List<Integer> graphSizes = train(model, optimizer, loss, data, batch, 1);

        assertEquals("one optimizer step", 1, optimizer.steps.size());
        Map<Parameter, float[]> applied = optimizer.steps.get(0);
        for (Map.Entry<String, Parameter> entry : model.namedParameters().entrySet()) {
            String name = api + " " + entry.getKey();
            Parameter p = entry.getValue();
            float[] g = applied.get(p);
            assertNotNull(name + " received a gradient", g);
            assertTrue(name + " gradient is not zero", hasNonZero(g));
            assertClose(name + " gradient equals a separate backward pass", reference.get(p), g, 1e-5f);
            float[] w0 = before.get(p);
            float[] w1 = p.tensor().toFloatArray();
            float[] stepped = new float[w0.length];
            for (int i = 0; i < w0.length; i++) {
                stepped[i] = w0[i] - LR * g[i];
            }
            assertTrue(name + " changed", changed(w0, w1));
            assertClose(name + " is the SGD update", stepped, w1, 1e-5f);
            assertTrue(name + " still requires grad", p.tensor().requiresGrad());
            assertNull(name + " is still a leaf", p.tensor().gradFn());
            assertNull(name + " gradient zeroed after the step", p.grad());
        }
        assertGraphEmptyAfterEveryBatch(graphSizes);
    }

    // ------------------------------------------------------------------ accumulation

    @Test
    public void accumulatedStepAppliesTheMeanOfTheBatchGradients() {
        Candidate<Mlp> mlp = kinkFree(seed -> new Mlp(backend, new Random(seed)), Shape.of(2, MLP_IN));
        Module model = mlp.model;
        LossFunction loss = new CrossEntropyLoss();
        float[] labels = labels(2);
        Map<Parameter, float[]> first = sampleGradients(model, loss, mlp.inputs, labels, 0);
        Map<Parameter, float[]> second = sampleGradients(model, loss, mlp.inputs, labels, 1);

        TensorDataset data = track(dataset(mlp.inputs, labels, Shape.of(MLP_IN), 2));
        RecordingOptimizer optimizer = new RecordingOptimizer(model.parameters(), LR);
        List<Integer> graphSizes = train(model, optimizer, loss, data, 1, 2);

        assertEquals("two batches, one step", 1, optimizer.steps.size());
        for (Map.Entry<String, Parameter> entry : model.namedParameters().entrySet()) {
            Parameter p = entry.getValue();
            float[] a = first.get(p);
            float[] b = second.get(p);
            float[] mean = new float[a.length];
            for (int i = 0; i < a.length; i++) {
                mean[i] = (a[i] + b[i]) * 0.5f;
            }
            assertClose(api + " " + entry.getKey() + " applied gradient is the mean", mean,
                optimizer.steps.get(0).get(p), 1e-5f);
        }
        assertGraphEmptyAfterEveryBatch(graphSizes);
    }

    /** The gradients of the loss of sample {@code index} alone; the parameters' gradient slots are left empty. */
    private Map<Parameter, float[]> sampleGradients(Module model, LossFunction loss, float[] inputs, float[] labels,
                                                    int index) {
        float[] row = new float[MLP_IN];
        System.arraycopy(inputs, index * MLP_IN, row, 0, MLP_IN);
        try (Tensor x = tensor(row, Shape.of(1, MLP_IN));
             Tensor y = tensor(new float[]{labels[index]}, Shape.of(1, 1))) {
            return backwardGradients(model, loss, x, y);
        }
    }

    // ------------------------------------------------------------------ auxiliary loss

    @Test
    public void mixtureOfExpertsRouterReceivesTheLoadBalanceGradient() {
        int tokens = 3;
        int dModel = 4;
        float[] inputs = new float[tokens * dModel];
        float[] lead = {0.3f, 0.55f, 0.8f};
        for (int t = 0; t < tokens; t++) {
            inputs[t * dModel] = lead[t];
            for (int j = 1; j < dModel; j++) {
                inputs[t * dModel + j] = 0.1f * j * (t + 1) - 0.2f;
            }
        }
        float[] withTerm = routerGradient(true, inputs, tokens, dModel);
        float[] withoutTerm = routerGradient(false, inputs, tokens, dModel);

        float[] expected = loadBalanceRouterGradient(inputs, lead, tokens, dModel);
        assertClose(api + " router weight and bias gradient of 2 * mean_t p0(t)", expected, withTerm, 1e-4f);
        assertTrue("the load-balance term gives the router a gradient", hasNonZero(withTerm));
        assertTrue("without the term the zero main loss gives the router none", !hasNonZero(withoutTerm));
    }

    /**
     * The router's weight gradient (row-major {@code [2, dModel]}) followed by its bias gradient, as applied by one
     * {@link Trainer} step of a two-expert top-1 mixture of experts under a main loss whose gradient is zero. The
     * router sends every token to expert 0 ({@code logits = (x_0, -x_0)} with {@code x_0 > 0}).
     */
    private float[] routerGradient(boolean loadBalancing, float[] inputs, int tokens, int dModel) {
        MixtureOfExperts moe = track(new MixtureOfExperts(dModel, 8, 2, 1, 0.0f, loadBalancing));
        moe.initialize(backend, new Random(7));
        Map<String, Parameter> named = moe.namedParameters();
        Parameter weight = named.get("router.weight");
        Parameter bias = named.get("router.bias");
        assertNotNull("router weight", weight);
        assertNotNull("router bias", bias);
        float[] w = new float[2 * dModel];
        w[0] = 1.0f;
        w[dModel] = -1.0f;
        try (Tensor routerWeight = tensor(w, Shape.of(2, dModel));
             Tensor routerBias = tensor(new float[2], Shape.of(2))) {
            weight.copyFrom(routerWeight);
            bias.copyFrom(routerBias);
        }

        TensorDataset data = track(dataset(inputs, new float[]{0.0f}, Shape.of(tokens, dModel), 1));
        RecordingOptimizer optimizer = new RecordingOptimizer(moe.parameters(), LR);
        List<Integer> graphSizes = train(moe, optimizer, new ZeroLoss(), data, 1, 1);
        assertGraphEmptyAfterEveryBatch(graphSizes);
        assertNull("no auxiliary term left pending", moe.takeAuxiliaryLoss());

        float[] gw = optimizer.steps.get(0).get(weight);
        float[] gb = optimizer.steps.get(0).get(bias);
        assertNotNull("the router weight received a gradient", gw);
        assertNotNull("the router bias received a gradient", gb);
        float[] out = new float[gw.length + gb.length];
        System.arraycopy(gw, 0, out, 0, gw.length);
        System.arraycopy(gb, 0, out, gw.length, gb.length);
        return out;
    }

    /**
     * The gradient of {@code 2 * sum_e meanProb_e * share_e} with every token on expert 0 ({@code share = (1, 0)}):
     * with {@code p0 = sigmoid(2 x_0)}, the logit gradients of token t are {@code (2/T) p0 (1 - p0)} and
     * {@code -(2/T) p0 (1 - p0)}; the weight gradient is their outer product with the token's input summed over
     * tokens, the bias gradient their sum.
     */
    private static float[] loadBalanceRouterGradient(float[] inputs, float[] lead, int tokens, int dModel) {
        double[] out = new double[2 * dModel + 2];
        for (int t = 0; t < tokens; t++) {
            double p0 = 1.0 / (1.0 + Math.exp(-2.0 * lead[t]));
            double g0 = 2.0 / tokens * p0 * (1.0 - p0);
            double[] logitGrad = {g0, -g0};
            for (int k = 0; k < 2; k++) {
                for (int j = 0; j < dModel; j++) {
                    out[k * dModel + j] += logitGrad[k] * inputs[t * dModel + j];
                }
                out[2 * dModel + k] += logitGrad[k];
            }
        }
        float[] result = new float[out.length];
        for (int i = 0; i < out.length; i++) {
            result[i] = (float) out[i];
        }
        return result;
    }

    // ------------------------------------------------------------------ training helpers

    /**
     * Trains {@code model} for one epoch over {@code data} with batches of {@code batchSize}; returns the graph size
     * after each batch.
     */
    private List<Integer> train(Module model, Optimizer optimizer, LossFunction loss, TensorDataset data,
                                int batchSize, int accumulationSteps) {
        List<Integer> graphSizes = new ArrayList<>();
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(loss)
            .trainLoader(new DataLoader<>(data, batchSize, false))
            .config(TrainerConfig.builder().accumulationSteps(accumulationSteps).verbose(false).build())
            .addCallback(new Callback() {
                @Override
                public void onBatchEnd(int batch, float batchLoss) {
                    graphSizes.add(ComputationGraph.current().size());
                }
            })
            .build();
        trainer.train(1);
        return graphSizes;
    }

    private static void assertGraphEmptyAfterEveryBatch(List<Integer> graphSizes) {
        assertTrue("at least one batch ran", !graphSizes.isEmpty());
        for (int i = 0; i < graphSizes.size(); i++) {
            assertEquals("graph size after batch " + i, 0, (int) graphSizes.get(i));
        }
        assertEquals("graph size after training", 0, ComputationGraph.current().size());
    }

    /**
     * The parameter gradients of one backward pass of {@code loss(model(x), y)} inside a step scope; the gradient slots
     * are emptied afterwards.
     */
    private static Map<Parameter, float[]> backwardGradients(Module model, LossFunction loss, Tensor x, Tensor y) {
        try (ComputationGraph.StepScope step = ComputationGraph.current().step()) {
            try (Tensor value = lossOf(model, loss, x, y)) {
                value.backward();
            }
        }
        Map<Parameter, float[]> grads = new IdentityHashMap<>();
        for (Parameter p : model.parameters()) {
            Tensor g = p.grad();
            grads.put(p, g == null ? null : g.toFloatArray());
            p.zeroGrad();
        }
        return grads;
    }

    /** {@code loss(model(x), y)} as a new tensor; the prediction is closed. */
    private static Tensor lossOf(Module model, LossFunction loss, Tensor x, Tensor y) {
        try (Tensor prediction = model.forward(x)) {
            return loss.compute(prediction, y);
        }
    }

    private static Map<Parameter, float[]> values(Module model) {
        Map<Parameter, float[]> values = new IdentityHashMap<>();
        for (Parameter p : model.parameters()) {
            values.put(p, p.tensor().toFloatArray());
        }
        return values;
    }

    /** Class labels {@code i % CLASSES} as floats. */
    private static float[] labels(int n) {
        float[] labels = new float[n];
        for (int i = 0; i < n; i++) {
            labels[i] = i % CLASSES;
        }
        return labels;
    }

    /** {@code n} samples: consecutive blocks of {@code inputs} of {@code sampleShape}, each with its label. */
    private TensorDataset dataset(float[] inputs, float[] labels, Shape sampleShape, int n) {
        int size = (int) sampleShape.numElements();
        Tensor[] in = new Tensor[n];
        Tensor[] out = new Tensor[n];
        for (int i = 0; i < n; i++) {
            float[] block = new float[size];
            System.arraycopy(inputs, i * size, block, 0, size);
            in[i] = tensor(block, sampleShape);
            out[i] = Tensor.fromArray(backend, new float[]{labels[i]});
        }
        return new TensorDataset(in, out);
    }

    private Tensor tensor(float[] data, Shape shape) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    private <T extends AutoCloseable> T track(T t) {
        owned.add(t);
        return t;
    }

    // ------------------------------------------------------------------ kink-free models

    /** A model and the input batch it was accepted with. */
    private static final class Candidate<M extends Module> {
        final M model;
        final float[] inputs;
        final Shape inputShape;

        Candidate(M model, float[] inputs, Shape inputShape) {
            this.model = model;
            this.inputs = inputs;
            this.inputShape = inputShape;
        }
    }

    /** A model that reports how far its relu inputs and max-pool choices are from a kink for an input batch. */
    interface KinkAware {
        /** The smallest distance from a kink over the batch, computed without recording. */
        float kinkMargin(Tensor input);
    }

    /**
     * The model built from the first seed whose inputs (uniform in {@code [-1, 1)} from the same seed) keep every kink
     * at least {@link #MARGIN} away; the rejected models are closed.
     */
    private <M extends Module & KinkAware> Candidate<M> kinkFree(IntFunction<M> build, Shape inputShape) {
        int n = (int) inputShape.numElements();
        for (int seed = 0; seed < SEED_SEARCH; seed++) {
            M model = build.apply(seed);
            float[] inputs = TrainParity.uniform(10_000L + seed, n, -1f, 1f);
            float margin;
            try (Tensor x = tensor(inputs, inputShape)) {
                margin = model.kinkMargin(x);
            } catch (RuntimeException e) {
                model.close();
                throw e;
            }
            if (margin >= MARGIN) {
                return new Candidate<>(track(model), inputs, inputShape);
            }
            model.close();
        }
        fail("no seed below " + SEED_SEARCH + " keeps every kink " + MARGIN + " away");
        throw new AssertionError("unreachable");
    }

    /** The MLP: {@code fc1 -> relu -> fc2 -> relu -> fc3}. */
    static final class Mlp extends Module implements KinkAware {
        private final Linear fc1;
        private final Linear fc2;
        private final Linear fc3;

        Mlp(ComputeBackend backend, Random random) {
            fc1 = linear("fc1", MLP_IN, MLP_HIDDEN, backend, random);
            fc2 = linear("fc2", MLP_HIDDEN, MLP_HIDDEN, backend, random);
            fc3 = linear("fc3", MLP_HIDDEN, CLASSES, backend, random);
        }

        private Linear linear(String name, int in, int out, ComputeBackend backend, Random random) {
            Linear layer = new Linear(in, out);
            layer.initialize(backend, random);
            registerModule(name, layer);
            return layer;
        }

        @Override
        public Tensor forward(Tensor input) {
            try (Tensor z1 = fc1.forward(input);
                 Tensor h1 = TensorOps.relu(z1);
                 Tensor z2 = fc2.forward(h1);
                 Tensor h2 = TensorOps.relu(z2)) {
                return fc3.forward(h2);
            }
        }

        @Override
        public float kinkMargin(Tensor input) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor z1 = fc1.forward(input);
                 Tensor h1 = TensorOps.relu(z1);
                 Tensor z2 = fc2.forward(h1)) {
                float relu = Math.min(minAbs(z1), minAbs(z2));
                return Math.min(relu, Math.min(max(z1), max(z2)));
            }
        }
    }

    /** The CNN: {@code conv1 -> relu -> pool1 -> conv2 -> relu -> pool2 -> reshape -> fc}. */
    static final class Cnn extends Module implements KinkAware {
        private final Conv2d conv1;
        private final Conv2d conv2;
        private final MaxPool2d pool1;
        private final MaxPool2d pool2;
        private final Linear fc;

        Cnn(ComputeBackend backend, Random random) {
            conv1 = conv("conv1", 1, backend, random);
            pool1 = registerModule("pool1", pool());
            conv2 = conv("conv2", CNN_CHANNELS, backend, random);
            pool2 = registerModule("pool2", pool());
            fc = new Linear(CNN_FLAT, CLASSES);
            fc.initialize(backend, random);
            registerModule("fc", fc);
        }

        private Conv2d conv(String name, int in, ComputeBackend backend, Random random) {
            Conv2d layer = new Conv2d(in, CNN_CHANNELS, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
            layer.initialize(backend, random);
            registerModule(name, layer);
            return layer;
        }

        private static MaxPool2d pool() {
            return new MaxPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0});
        }

        @Override
        public Tensor forward(Tensor input) {
            int batch = input.shape().dim(0);
            try (Tensor z1 = conv1.forward(input);
                 Tensor h1 = TensorOps.relu(z1);
                 Tensor p1 = pool1.forward(h1);
                 Tensor z2 = conv2.forward(p1);
                 Tensor h2 = TensorOps.relu(z2);
                 Tensor p2 = pool2.forward(h2);
                 Tensor flat = p2.reshape(batch, CNN_FLAT)) {
                return fc.forward(flat);
            }
        }

        @Override
        public float kinkMargin(Tensor input) {
            int batch = input.shape().dim(0);
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor z1 = conv1.forward(input);
                 Tensor h1 = TensorOps.relu(z1);
                 Tensor p1 = pool1.forward(h1);
                 Tensor z2 = conv2.forward(p1);
                 Tensor h2 = TensorOps.relu(z2)) {
                float[] second = h2.toFloatArray();
                float relu = Math.min(minAbs(z1), minAbs(z2));
                float pools = Math.min(poolGap(h1.toFloatArray(), batch, CNN_SIZE),
                    poolGap(second, batch, CNN_SIZE / 2));
                return Math.min(Math.min(relu, pools), smallestWindowMax(second, batch, CNN_SIZE / 2));
            }
        }

        /**
         * The smallest window maximum of a 2x2 stride-2 pooling of {@code [batch, CNN_CHANNELS, size, size]} values:
         * positive when every pooled feature the classifier reads is alive, so the gradient reaches every layer.
         */
        private static float smallestWindowMax(float[] values, int batch, int size) {
            float smallest = Float.POSITIVE_INFINITY;
            for (int plane = 0; plane < batch * CNN_CHANNELS; plane++) {
                for (int oh = 0; oh + 1 < size; oh += 2) {
                    for (int ow = 0; ow + 1 < size; ow += 2) {
                        float max = Float.NEGATIVE_INFINITY;
                        for (int dh = 0; dh < 2; dh++) {
                            for (int dw = 0; dw < 2; dw++) {
                                max = Math.max(max, values[(plane * size + oh + dh) * size + ow + dw]);
                            }
                        }
                        smallest = Math.min(smallest, max);
                    }
                }
            }
            return smallest;
        }

        /**
         * The smallest gap between the largest and the second largest value of a 2x2 stride-2 window of
         * {@code [batch, CNN_CHANNELS, size, size]} values, over the windows whose largest value is positive (a window
         * of relu zeros has no choice to flip).
         */
        private static float poolGap(float[] values, int batch, int size) {
            float gap = Float.POSITIVE_INFINITY;
            for (int plane = 0; plane < batch * CNN_CHANNELS; plane++) {
                for (int oh = 0; oh + 1 < size; oh += 2) {
                    for (int ow = 0; ow + 1 < size; ow += 2) {
                        float first = Float.NEGATIVE_INFINITY;
                        float second = Float.NEGATIVE_INFINITY;
                        for (int dh = 0; dh < 2; dh++) {
                            for (int dw = 0; dw < 2; dw++) {
                                float v = values[(plane * size + oh + dh) * size + ow + dw];
                                if (v > first) {
                                    second = first;
                                    first = v;
                                } else if (v > second) {
                                    second = v;
                                }
                            }
                        }
                        if (first > 0) {
                            gap = Math.min(gap, first - second);
                        }
                    }
                }
            }
            return gap;
        }
    }

    /** The largest value of {@code t}: positive when some relu of it passes a gradient. */
    private static float max(Tensor t) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : t.toFloatArray()) {
            max = Math.max(max, v);
        }
        return max;
    }

    private static float minAbs(Tensor t) {
        float min = Float.POSITIVE_INFINITY;
        for (float v : t.toFloatArray()) {
            min = Math.min(min, Math.abs(v));
        }
        return min;
    }

    // ------------------------------------------------------------------ fixtures

    /** {@code sum(prediction * 0)}: a recorded loss whose gradient is zero everywhere. */
    static final class ZeroLoss implements LossFunction {
        @Override
        public Tensor compute(Tensor prediction, Tensor target) {
            try (Tensor zero = TensorOps.mul(prediction, 0.0f)) {
                return TensorOps.sum(zero);
            }
        }

        @Override
        public String name() {
            return "zero";
        }
    }

    /** SGD that records, at each step, a copy of the gradient it applies to every parameter (null for none). */
    static final class RecordingOptimizer extends Optimizer {
        private final SGD sgd;
        final List<Map<Parameter, float[]>> steps = new ArrayList<>();

        RecordingOptimizer(List<Parameter> params, float lr) {
            super(params, Collections.singletonMap("lr", lr));
            this.sgd = new SGD(params, lr);
        }

        @Override
        public void step() {
            Map<Parameter, float[]> grads = new IdentityHashMap<>();
            for (Parameter p : params) {
                Tensor g = p.grad();
                grads.put(p, g == null ? null : g.toFloatArray());
            }
            steps.add(grads);
            sgd.step();
        }
    }

    // ------------------------------------------------------------------ assertions

    private static boolean hasNonZero(float[] values) {
        for (float v : values) {
            if (v != 0.0f) {
                return true;
            }
        }
        return false;
    }

    private static boolean changed(float[] before, float[] after) {
        for (int i = 0; i < before.length; i++) {
            if (Float.floatToIntBits(before[i]) != Float.floatToIntBits(after[i])) {
                return true;
            }
        }
        return false;
    }

    /** {@code |expected - actual| <= tolerance * max(1, |expected|)} element by element. */
    private static void assertClose(String what, float[] expected, float[] actual, float tolerance) {
        assertNotNull(what + ": expected values", expected);
        assertNotNull(what + ": actual values", actual);
        assertEquals(what + ": length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            float allowed = tolerance * Math.max(1.0f, Math.abs(expected[i]));
            if (!(Math.abs(expected[i] - actual[i]) <= allowed)) {
                fail(what + ": element " + i + " expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }
}
