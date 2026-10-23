package com.eonmux.neuralfree.nn.vision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
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
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of the vision modules: for every module class with
 * parameters, the gradients of {@code sum(output * probe)} with respect to the module's input
 * ({@link GradientCheckUtil#gradCheck}) and to its parameters ({@link GradientCheckUtil#gradCheckParameters}) match
 * central differences with step 1e-3 within 1e-2, on the host reference backend and on every available GPU API
 * ({@link TrainParity}). The probe is a fixed tensor in [-1, 1], so every output element carries its own weight. The
 * modules run in training mode, so batch normalization normalizes with the batch statistics and its full backward is
 * checked; the parameter check restores the running statistics after every evaluation.
 *
 * <p><b>Values.</b> Every weight of rank 2 or more is uniform in {@code +-1/sqrt(fanIn)} and every other non-norm
 * parameter uniform in [-0.1, 0.1], from seeds; inputs are uniform in [-1, 1] unless a test builds them.</p>
 *
 * <p><b>Kinks: ReLU.</b> In training mode a normalized value satisfies {@code |x_hat| <= sqrt(n - 1)} for the
 * {@code n} values of its channel (biased variance). Every batch norm gets {@code gamma} in [0.2, 0.3] and
 * {@code beta >= 0.3 * sqrt(maxCount - 1) + 0.5 + shortcutBound}, where {@code maxCount} bounds {@code n} over the
 * module and {@code shortcutBound} bounds an identity shortcut added before a ReLU. Every ReLU input is then at least
 * 0.5, every ReLU acts as the identity on the whole checked neighbourhood, and the network is smooth. The
 * squeeze-and-excitation ReLU sees pooled values of at least 0.5 through a first layer whose rows are all positive or
 * all negative (entries of magnitude 0.1 to 0.5, biases below 0.05), so its inputs stay at least 0.35 away from 0.</p>
 *
 * <p><b>Kinks: max pooling.</b> The convolutions in front of a max pool are set to copy one input channel through
 * their centre tap (every other tap 0), so after the order-preserving batch norm and identity ReLU the values in each
 * pooling window are ordered as the input values they copy. The tests build inputs whose values inside every pooling
 * window differ by at least 0.13 (ResNet) or 0.3 (UNet). Perturbing an input element, a centre tap, a bias or a norm
 * parameter keeps that order (it moves one value by 1e-3 of input, scales or shifts a whole channel, or changes a
 * monotone map), and perturbing a tap that is 0 moves each pooled value by less than a fifth of the spacing (1e-3
 * times an input value or a bounded batch-norm output), so no evaluation changes a window's maximum. The taps that
 * are 0 still receive their analytical gradient.</p>
 *
 * <p><b>Lifetime.</b> Input checks run with the module frozen, so no parameter slot is written; parameter checks
 * restore every slot. Each check leaves the thread's graph at the size it had before the check.</p>
 */
@RunWith(Parameterized.class)
public class VisionGradCheckTest {

    private static final long SEED = 5151L;
    private static final float EPSILON = 1e-3f;
    private static final float TOLERANCE = 1e-2f;
    private static final int MAX_ELEMENTS = 3;
    private static final float GAMMA_MIN = 0.2f;
    private static final float GAMMA_MAX = 0.3f;
    private static final float RELU_MARGIN = 0.5f;
    /** Bound of an identity shortcut holding an input uniform in [-1, 1], perturbed by at most 1e-3. */
    private static final float INPUT_BOUND = 1.01f;

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

    public VisionGradCheckTest(String name, TrainParity.Api api) {
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

    // ------------------------------------------------------------------ blocks

    /** An identity shortcut (the raw input reaches the final ReLU), then a strided projection shortcut. */
    @Test
    public void residualBlock() {
        ResidualBlock identity = keep(new ResidualBlock(4, 4));
        identity.initialize(backend, new Random(SEED));
        prepare(identity, 2 * 4 * 4, INPUT_BOUND);
        checkModule("ResidualBlock identity", identity, tensor(11, 2, 4, 4, 4), 12, 2, 4, 4, 4);

        ResidualBlock projected = keep(new ResidualBlock(4, 6, new int[]{2, 2}));
        projected.initialize(backend, new Random(SEED));
        prepare(projected, 2 * 2 * 2, 0f);
        checkModule("ResidualBlock projection", projected, tensor(13, 2, 4, 4, 4), 14, 2, 6, 2, 2);
    }

    /** A projection with squeeze-and-excitation, then a strided projection without it. */
    @Test
    public void bottleneckBlock() {
        BottleneckBlock squeezed = keep(new BottleneckBlock(4, 8, 8, new int[]{1, 1}, true, 4));
        squeezed.initialize(backend, new Random(SEED));
        prepare(squeezed, 2 * 4 * 4, 0f);
        Parameter fc1 = squeezed.getParameter("se_fc1.weight");
        assertNotNull("squeeze-and-excitation first layer", fc1);
        int rows = fc1.shape().dim(0);
        int cols = fc1.shape().dim(1);
        float[] signed = new float[rows * cols];
        float[] magnitudes = TrainParity.uniform(21, rows * cols, 0.1f, 0.5f);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                signed[r * cols + c] = (r % 2 == 0 ? 1f : -1f) * magnitudes[r * cols + c];
            }
        }
        store(fc1, signed);
        Parameter fc1Bias = squeezed.getParameter("se_fc1.bias");
        assertNotNull("squeeze-and-excitation first bias", fc1Bias);
        store(fc1Bias, TrainParity.uniform(22, rows, -0.05f, 0.05f));
        checkModule("BottleneckBlock squeeze-excitation", squeezed, tensor(23, 2, 4, 4, 4), 24, 2, 8, 4, 4);

        BottleneckBlock strided = keep(new BottleneckBlock(8, 4, 8, new int[]{2, 2}));
        strided.initialize(backend, new Random(SEED));
        prepare(strided, 2 * 4 * 4, 0f);
        checkModule("BottleneckBlock strided", strided, tensor(25, 2, 8, 4, 4), 26, 2, 8, 2, 2);
    }

    @Test
    public void depthwiseSeparableConv() {
        DepthwiseSeparableConv conv = keep(new DepthwiseSeparableConv(4, 6, new int[]{3, 3}, new int[]{1, 1},
            new int[]{1, 1}, true, true));
        conv.initialize(backend);
        prepare(conv, 2 * 4 * 4, 0f);
        checkModule("DepthwiseSeparableConv", conv, tensor(31, 2, 4, 4, 4), 32, 2, 6, 4, 4);
    }

    // ------------------------------------------------------------------ networks

    /**
     * ResNet-18 on {@code [4, 3, 8, 8]} (the deepest stages at 1x1, four values per channel). The stem copies input
     * channel {@code o % 3} into channel {@code o} through its centre tap, which reads input position
     * {@code (2 oh, 2 ow)}; the input holds, per batch row and channel, a permutation of 16 values spaced 2/15 apart at
     * the even positions, so the stem's max pool has well-separated window maxima. The parameters checked are one or
     * two of every kind along the network: stem, identity and projection blocks, and the classifier. Two elements are
     * checked per parameter, evenly spaced in row-major order, which for a 3x3 kernel are both tap (0, 0): the 3x3
     * weights checked are therefore those of the first stage, whose 2x2 maps give that tap a live input, while the
     * 1x1 maps of the later stages are checked through their 1x1 projection weights.
     * The input gradient is not checked here: it enters through the stem convolution, whose weight gradient is, and
     * the residual blocks' input gradients are checked by {@link #residualBlock()}.
     */
    @Test
    public void resNet() {
        ResNet resnet = keep(ResNet.resnet18(5));
        resnet.initialize(backend, new Random(SEED));
        prepare(resnet, 4 * 4 * 4, 0f);
        Parameter stem = resnet.getParameter("conv1.weight");
        store(stem, centreTapCopy(64, 3, 7, (o, i) -> i == o % 3));

        int batch = 4;
        float[] x = TrainParity.uniform(41, batch * 3 * 8 * 8, -1f, 1f);
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < 3; c++) {
                for (int k = 0; k < 16; k++) {
                    int h = 2 * (k / 4);
                    int w = 2 * (k % 4);
                    x[((b * 3 + c) * 8 + h) * 8 + w] = -1f + 2f * ((k * 7 + b * 3 + c * 5) % 16) / 15f;
                }
            }
        }
        Tensor input = keep(upload(x, batch, 3, 8, 8));
        Tensor probe = tensor(42, batch, 5);

        List<Parameter> checked = new ArrayList<>();
        for (String parameter : new String[]{"conv1.weight", "bn1.weight", "bn1.bias", "layers.0.conv1.weight",
            "layers.1.bn2.weight", "layers.2.projection.weight", "layers.2.projection_bn.bias",
            "layers.4.projection.weight", "layers.7.bn1.weight", "fc.weight", "fc.bias"}) {
            Parameter p = resnet.getParameter(parameter);
            assertNotNull("ResNet parameter " + parameter, p);
            checked.add(p);
        }
        checkParameters("ResNet-18", resnet, checked, 2, () -> probeSum(resnet.forward(input), probe));
    }

    /**
     * A two-level UNet on {@code [2, 1, 9, 9]}: the 9x9 skip is cropped to the 8x8 upsampled map, so the crop's slice
     * views are on the path. The first encoder block copies the input through the centre taps of both convolutions;
     * the input gives the four values of every 2x2 pooling window (rows and columns 0..7) spacings of 0.3, so the
     * encoder's max pool keeps its window maxima. Every parameter and the input are checked.
     */
    @Test
    public void uNet() {
        UNet unet = keep(new UNet(1, 2, new int[]{2, 4}, true));
        unet.initialize(backend, new Random(SEED));
        prepare(unet, 2 * 9 * 9, 0f);
        List<Conv2d> firstBlock = new ArrayList<>();
        for (Module layer : ((Sequential) unet.encoderBlocks().get(0)).layers()) {
            if (layer instanceof Conv2d) {
                firstBlock.add((Conv2d) layer);
            }
        }
        assertEquals("the first encoder block has two convolutions", 2, firstBlock.size());
        store(firstBlock.get(0).weight(), centreTapCopy(2, 1, 3, (o, i) -> true));
        store(firstBlock.get(1).weight(), centreTapCopy(2, 2, 3, (o, i) -> o == i));

        float[] x = TrainParity.uniform(51, 2 * 9 * 9, -1f, 1f);
        for (int b = 0; b < 2; b++) {
            for (int h = 0; h < 8; h++) {
                for (int w = 0; w < 8; w++) {
                    int window = (h / 2) * 4 + (w / 2);
                    int slot = (h % 2) * 2 + (w % 2);
                    float base = -0.95f + 0.9f * ((window * 5 + b * 3) % 16) / 15f;
                    x[(b * 9 + h) * 9 + w] = base + 0.3f * ((slot + window) % 4);
                }
            }
        }
        checkModule("UNet", unet, keep(upload(x, 2, 1, 9, 9)), 52, 2, 2, 8, 8);
    }

    // ------------------------------------------------------------------ checks

    /** The input check and the check of every parameter of {@code module}, for an output of {@code outDims}. */
    private void checkModule(String what, Module module, Tensor input, long probeSeed, int... outDims) {
        Tensor probe = tensor(probeSeed, outDims);
        checkInputs(what, module, in -> probeSum(module.forward(in.get(0)), probe), input);
        checkParameters(what, module, module.parameters(), MAX_ELEMENTS, () -> probeSum(module.forward(input), probe));
    }

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

    /** The gradients of {@code params} of {@code module} for {@code loss}, at most {@code maxElements} each. */
    private void checkParameters(String what, Module module, List<Parameter> params, int maxElements,
                                 Supplier<Tensor> loss) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(loss, module, params,
            EPSILON, TOLERANCE, maxElements);
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
     * Puts {@code module} in training mode, seeds every parameter and applies the batch-norm rule of the class
     * javadoc: {@code maxCount} bounds the values per channel of every batch norm, {@code shortcutBound} the magnitude
     * of an identity shortcut added before a ReLU.
     */
    private void prepare(Module module, int maxCount, float shortcutBound) {
        module.train();
        List<BatchNorm2d> norms = new ArrayList<>();
        Set<Parameter> normParameters = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Module m : module.modules()) {
            if (m instanceof BatchNorm2d) {
                BatchNorm2d bn = (BatchNorm2d) m;
                norms.add(bn);
                normParameters.add(bn.weight());
                normParameters.add(bn.bias());
            }
        }
        long seed = SEED;
        for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
            Parameter p = e.getValue();
            if (normParameters.contains(p)) {
                continue;
            }
            int n = (int) p.numElements();
            float bound = p.shape().rank() >= 2 ? (float) (1.0 / Math.sqrt(n / p.shape().dim(0))) : 0.1f;
            store(p, TrainParity.uniform(seed++, n, -bound, bound));
        }
        float betaFloor = (float) (GAMMA_MAX * Math.sqrt(maxCount - 1)) + RELU_MARGIN + shortcutBound;
        for (BatchNorm2d bn : norms) {
            int c = bn.numFeatures();
            store(bn.weight(), TrainParity.uniform(seed++, c, GAMMA_MIN, GAMMA_MAX));
            store(bn.bias(), TrainParity.uniform(seed++, c, betaFloor, betaFloor + 0.25f));
        }
    }

    /**
     * Square {@code k x k} convolution weights {@code [out, in, k, k]} that are 1 at the centre tap of every
     * {@code (o, i)} pair {@code copies} accepts and 0 elsewhere.
     */
    private static float[] centreTapCopy(int out, int in, int k, CopyRule copies) {
        float[] w = new float[out * in * k * k];
        int centre = k / 2;
        for (int o = 0; o < out; o++) {
            for (int i = 0; i < in; i++) {
                if (copies.copies(o, i)) {
                    w[((o * in + i) * k + centre) * k + centre] = 1.0f;
                }
            }
        }
        return w;
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

    /** Which {@code (output, input)} channel pairs a centre-tap copy connects. */
    private interface CopyRule {
        boolean copies(int out, int in);
    }
}
