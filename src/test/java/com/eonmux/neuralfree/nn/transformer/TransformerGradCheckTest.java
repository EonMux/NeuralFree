package com.eonmux.neuralfree.nn.transformer;

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
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of the transformer modules: for every module class with
 * parameters, the gradients of {@code sum(output * probe)} with respect to the module's inputs
 * ({@link GradientCheckUtil#gradCheck}) and to its parameters ({@link GradientCheckUtil#gradCheckParameters}) match
 * central differences with step 1e-3 within 1e-2, on the host reference backend and on every available GPU API
 * ({@link TrainParity}). The probe is a fixed tensor in [-1, 1], so every output element carries its own weight. The
 * stacks run with the parameter-free sinusoidal and rotary encodings, so those paths are checked as well.
 *
 * <p><b>Inputs and determinism.</b> Inputs, memories and probes are uniform in [-1, 1] from fixed seeds and the
 * modules are built from a seeded {@link Random}, with every bias given non-zero seeded values. Dropout rates are 0 and
 * the modules run in evaluation mode, so every evaluation computes the same function. The mixture of experts runs in
 * training mode, the only mode in which its forward hands over its load-balance term; it has no dropout and no other
 * stochastic step, so it is deterministic as well.</p>
 *
 * <p><b>Kinks.</b> Attention, softmax, layer and RMS normalization, GELU, GLU and GeGLU are smooth; masked attention
 * weights are exactly 0 on both sides of a difference. The one discontinuity is the mixture of experts' top-k routing:
 * its test sets the router to {@code 2 I} plus offsets below 0.05 and builds every token from a dominant feature 1, a
 * second feature 0.5 and offsets below 0.05, so the first, second and remaining router logits are separated by more
 * than 0.5 while a perturbation moves a logit by at most about 3e-3, and no evaluation changes the routing.</p>
 *
 * <p><b>Lifetime.</b> Input checks run with the module frozen, so no parameter slot is written; parameter checks
 * restore every slot. Each check leaves the thread's graph at the size it had before the check.</p>
 */
@RunWith(Parameterized.class)
public class TransformerGradCheckTest {

    private static final long SEED = 7373L;
    private static final float EPSILON = 1e-3f;
    private static final float TOLERANCE = 1e-2f;
    private static final int MAX_ELEMENTS = 4;
    private static final int D_MODEL = 8;
    private static final int HEADS = 2;
    private static final int D_FF = 16;

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

    public TransformerGradCheckTest(String name, TrainParity.Api api) {
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

    // ------------------------------------------------------------------ attention and blocks

    /**
     * Cross-attention with separate query, key and value inputs and a padding mask that excludes one key per batch
     * row, then causal self-attention.
     */
    @Test
    public void multiHeadAttention() {
        MultiHeadAttention cross = keep(new MultiHeadAttention(D_MODEL, HEADS, 0f, false, true));
        prepare(cross, (b, r) -> cross.initialize(b, r));
        Tensor q = tensor(11, 2, 3, D_MODEL);
        Tensor k = tensor(12, 2, 4, D_MODEL);
        Tensor v = tensor(13, 2, 4, D_MODEL);
        Tensor mask = keep(upload(new float[]{0f, 0.25f, Float.NEGATIVE_INFINITY, -0.5f,
            Float.NEGATIVE_INFINITY, 0f, 0.5f, 0f}, 2, 1, 1, 4));
        Tensor crossProbe = tensor(14, 2, 3, D_MODEL);
        checkInputs("cross attention", cross,
            in -> probeSum(cross.forward(in.get(0), in.get(1), in.get(2), mask), crossProbe), q, k, v);
        checkParameters("cross attention", cross, () -> probeSum(cross.forward(q, k, v, mask), crossProbe));

        MultiHeadAttention causal = keep(new MultiHeadAttention(D_MODEL, HEADS, 0f, true, true));
        prepare(causal, (b, r) -> causal.initialize(b, r));
        Tensor x = tensor(15, 2, 3, D_MODEL);
        Tensor selfProbe = tensor(16, 2, 3, D_MODEL);
        checkInputs("causal self-attention", causal, in -> probeSum(causal.forward(in.get(0)), selfProbe), x);
        checkParameters("causal self-attention", causal, () -> probeSum(causal.forward(x), selfProbe));
    }

    /** Pre-norm LayerNorm with the GELU MLP, then post-norm RMSNorm with GeGLU and causal attention. */
    @Test
    public void transformerEncoderLayer() {
        TransformerEncoderLayer preLn = keep(new TransformerEncoderLayer(D_MODEL, HEADS, D_FF, 0f, true,
            TransformerEncoderLayer.NormType.LAYER_NORM, TransformerEncoderLayer.FFNType.MLP, false));
        prepare(preLn, (b, r) -> preLn.initialize(b, r));
        checkSequenceBlock("pre-norm encoder layer", preLn, 21);

        TransformerEncoderLayer postRms = keep(new TransformerEncoderLayer(D_MODEL, HEADS, D_FF, 0f, false,
            TransformerEncoderLayer.NormType.RMS_NORM, TransformerEncoderLayer.FFNType.GEGLU, true));
        prepare(postRms, (b, r) -> postRms.initialize(b, r));
        checkSequenceBlock("post-norm causal encoder layer", postRms, 23);
    }

    /** Pre-norm LayerNorm with GLU, then post-norm RMSNorm with the GELU MLP; both attend over a memory input. */
    @Test
    public void transformerDecoderLayer() {
        TransformerDecoderLayer preLn = keep(new TransformerDecoderLayer(D_MODEL, HEADS, D_FF, 0f, true,
            TransformerDecoderLayer.NormType.LAYER_NORM, TransformerDecoderLayer.FFNType.GLU));
        prepare(preLn, (b, r) -> preLn.initialize(b, r));
        checkDecoder("pre-norm decoder layer", preLn, (x, m) -> preLn.forward(x, m), 31);

        TransformerDecoderLayer postRms = keep(new TransformerDecoderLayer(D_MODEL, HEADS, D_FF, 0f, false,
            TransformerDecoderLayer.NormType.RMS_NORM, TransformerDecoderLayer.FFNType.MLP));
        prepare(postRms, (b, r) -> postRms.initialize(b, r));
        checkDecoder("post-norm decoder layer", postRms, (x, m) -> postRms.forward(x, m), 33);
    }

    /** Two pre-norm layers with the final LayerNorm, after the sinusoidal position table. */
    @Test
    public void transformerEncoder() {
        TransformerEncoder encoder = keep(new TransformerEncoder(2, D_MODEL, HEADS, D_FF, 0f, true,
            TransformerEncoderLayer.NormType.LAYER_NORM, TransformerEncoderLayer.FFNType.MLP, false));
        prepare(encoder, (b, r) -> encoder.initialize(b, r));
        SinusoidalPE pe = new SinusoidalPE(D_MODEL, 6);
        owned.add(pe::close);
        pe.initialize(backend);
        encoder.setPositionalEncoding(pe);
        checkSequenceBlock("encoder stack", encoder, 41);
    }

    /** Two post-norm RMSNorm GeGLU layers after a rotary encoding of half the model width. */
    @Test
    public void transformerDecoder() {
        TransformerDecoder decoder = keep(new TransformerDecoder(2, D_MODEL, HEADS, D_FF, 0f, false,
            TransformerDecoderLayer.NormType.RMS_NORM, TransformerDecoderLayer.FFNType.GEGLU));
        prepare(decoder, (b, r) -> decoder.initialize(b, r));
        RotaryPE pe = new RotaryPE(D_MODEL, 6, D_MODEL / 2, 10000.0f);
        owned.add(pe::close);
        pe.initialize(backend);
        decoder.setPositionalEncoding(pe);
        checkDecoder("decoder stack", decoder, (x, m) -> decoder.forward(x, m), 51);
    }

    /** The table rows at positions 2..4 receive the gradient; the other rows' gradient is 0 on both sides. */
    @Test
    public void learnedPositionalEncoding() {
        LearnedPE pe = keep(new LearnedPE(D_MODEL, 6));
        prepare(pe, (b, r) -> pe.initialize(b, r));
        Tensor x = tensor(61, 2, 3, D_MODEL);
        Tensor probe = tensor(62, 2, 3, D_MODEL);
        checkInputs("learned position table", pe, in -> probeSum(pe.apply(in.get(0), 2), probe), x);
        checkParameters("learned position table", pe, 6 * D_MODEL, () -> probeSum(pe.apply(x, 2), probe));
    }

    // ------------------------------------------------------------------ mixture of experts

    /**
     * Top-2 of 4 experts with load balancing: the checked sum adds the auxiliary load-balance term the forward hands
     * over ({@link Module#takeAuxiliaryLoss()}), so the router is checked through the gates and through that term.
     * The module runs in training mode, since a forward in evaluation mode adds no auxiliary term. The routing is
     * fixed by construction (class javadoc).
     */
    @Test
    public void mixtureOfExperts() {
        int dModel = 4;
        MixtureOfExperts moe = keep(new MixtureOfExperts(dModel, 6, 4, 2));
        prepare(moe, (b, r) -> moe.initialize(b, r));
        moe.train();
        float[] router = TrainParity.uniform(71, dModel * dModel, -0.05f, 0.05f);
        for (int e = 0; e < dModel; e++) {
            router[e * dModel + e] = 2.0f;
        }
        store(moe.router().weight(), router);
        store(moe.router().bias(), TrainParity.uniform(72, dModel, -0.05f, 0.05f));

        int[][] routes = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}, {1, 3}};
        float[] tokens = TrainParity.uniform(73, routes.length * dModel, -0.05f, 0.05f);
        for (int t = 0; t < routes.length; t++) {
            tokens[t * dModel + routes[t][0]] = 1.0f;
            tokens[t * dModel + routes[t][1]] = 0.5f;
        }
        Tensor x = keep(upload(tokens, 2, 3, dModel));
        Tensor probe = tensor(74, 2, 3, dModel);
        checkInputs("mixture of experts", moe, in -> moeLoss(moe, in.get(0), probe), x);
        checkParameters("mixture of experts", moe, MAX_ELEMENTS, () -> moeLoss(moe, x, probe));
    }

    /** {@code sum(moe(x) * probe)} plus the auxiliary load-balance term of that forward. */
    private static Tensor moeLoss(MixtureOfExperts moe, Tensor x, Tensor probe) {
        try (Tensor main = probeSum(moe.forward(x), probe);
             Tensor aux = moe.takeAuxiliaryLoss()) {
            assertNotNull("load balancing hands over an auxiliary term", aux);
            return TensorOps.add(main, aux);
        }
    }

    // ------------------------------------------------------------------ shared cases

    /** A block mapping {@code [2, 3, D_MODEL]} to the same shape. */
    private void checkSequenceBlock(String what, Module block, long seed) {
        Tensor x = tensor(seed, 2, 3, D_MODEL);
        Tensor probe = tensor(seed + 1, 2, 3, D_MODEL);
        checkInputs(what, block, in -> probeSum(block.forward(in.get(0)), probe), x);
        checkParameters(what, block, MAX_ELEMENTS, () -> probeSum(block.forward(x), probe));
    }

    /** A decoder over input {@code [2, 3, D_MODEL]} and memory {@code [2, 4, D_MODEL]}; both inputs are checked. */
    private void checkDecoder(String what, Module decoder, DecoderCall call, long seed) {
        Tensor x = tensor(seed, 2, 3, D_MODEL);
        Tensor memory = tensor(seed + 1, 2, 4, D_MODEL);
        Tensor probe = tensor(seed + 2, 2, 3, D_MODEL);
        checkInputs(what, decoder, in -> probeSum(call.forward(in.get(0), in.get(1)), probe), x, memory);
        checkParameters(what, decoder, MAX_ELEMENTS, () -> probeSum(call.forward(x, memory), probe));
    }

    // ------------------------------------------------------------------ checks

    /** The input gradients of {@code fn} at {@code inputs}, with {@code module}'s parameters frozen meanwhile. */
    private void checkInputs(String what, Module module, GradientCheckUtil.TensorFunction fn, Tensor... inputs) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result;
        module.setRequiresGrad(false);
        try {
            result = GradientCheckUtil.gradCheck(fn, Arrays.asList(inputs), EPSILON, TOLERANCE);
        } finally {
            module.setRequiresGrad(true);
        }
        assertTrue(name + " " + what + " input gradients: " + result, result.passed());
        assertEquals(name + " " + what + " input check leaves the graph as it was", before, graph.size());
    }

    /** The gradients of every parameter of {@code module} for {@code loss}, at most {@code maxElements} each. */
    private void checkParameters(String what, Module module, int maxElements, Supplier<Tensor> loss) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(loss, module,
            module.parameters(), EPSILON, TOLERANCE, maxElements);
        assertTrue(name + " " + what + " parameter gradients: " + result, result.passed());
        assertEquals(name + " " + what + " parameter check leaves the graph as it was", before, graph.size());
    }

    private void checkParameters(String what, Module module, Supplier<Tensor> loss) {
        checkParameters(what, module, MAX_ELEMENTS, loss);
    }

    /** {@code sum(output * probe)} as a new scalar; {@code output} is closed. */
    static Tensor probeSum(Tensor output, Tensor probe) {
        try (Tensor out = output;
             Tensor weighted = TensorOps.mul(out, probe)) {
            return TensorOps.sum(weighted);
        }
    }

    // ------------------------------------------------------------------ setup

    /** Initializes a module on the backend from the seed, gives its biases seeded values and puts it in eval mode. */
    private void prepare(Module module, Initializer init) {
        init.initialize(backend, new Random(SEED));
        for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
            if (e.getKey().endsWith("bias")) {
                Parameter p = e.getValue();
                store(p, TrainParity.uniform(SEED + e.getKey().hashCode(), (int) p.numElements(), -0.5f, 0.5f));
            }
        }
        module.eval();
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

    /** One decoder application to an input and a memory. */
    private interface DecoderCall {
        Tensor forward(Tensor input, Tensor memory);
    }
}
