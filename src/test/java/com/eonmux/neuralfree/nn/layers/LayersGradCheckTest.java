package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.activations.Tanh;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.Collections;
import java.util.Random;

import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.distinct;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.in;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.uniform;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Gradient checks of the nn layers: the gradients a recorded backward pass gives a layer's input
 * ({@link GradCheck#inputs}) and its parameters ({@link GradCheck#parameters}) against central finite differences, on
 * the host reference backend and on every available GPU API. The parameterized layers are {@link Linear},
 * {@link Embedding}, {@link Conv2d}, {@link Conv1d} and {@link ConvTranspose2d}; the parameter-free layers
 * ({@link Flatten}, {@link Dropout}, {@link MaxPool2d}, {@link AvgPool2d}, {@link AdaptiveAvgPool2d}) are checked
 * on their input.
 *
 * <p>Weights come from each layer's seeded initialization (biases are overwritten with non-zero values). Kinks: max
 * pooling reads distinct grid values ({@link GradCheck#distinct}); the composed network uses tanh and average
 * pooling, which are smooth. Dropout is checked in training mode with its seed reset before every evaluation, so every
 * evaluation applies the same mask. The embedding's padding row, which by definition gets no gradient although the
 * forward copies it, is checked exactly instead of against finite differences.</p>
 */
@RunWith(Parameterized.class)
public class LayersGradCheckTest {

    private static final long INIT_SEED = 4321L;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return GradCheck.targets();
    }

    private final GradCheck.Target target;
    private GradCheck fx;

    public LayersGradCheckTest(GradCheck.Target target) {
        this.target = target;
    }

    @Before
    public void setUp() {
        fx = GradCheck.open(target);
    }

    @After
    public void tearDown() {
        if (fx != null) {
            fx.close();
        }
    }

    @Test
    public void linear() {
        Linear layer = fx.keep(new Linear(4, 5));
        layer.initialize(fx.backend(), new Random(INIT_SEED));
        fx.write(layer.bias().tensor(), uniform(301, 5, -0.5f, 0.5f));
        fx.inputs("Linear input", a -> layer.forward(a.get(0)), in(uniform(302, 24, -1f, 1f), 2, 3, 4));
        Tensor x = fx.constant(uniform(303, 8, -1f, 1f), 2, 4);
        fx.parameters("Linear", layer, () -> layer.forward(x));
    }

    @Test
    public void linearWithoutBiasOnAVector() {
        Linear layer = fx.keep(new Linear(4, 3, false));
        layer.initialize(fx.backend(), new Random(INIT_SEED));
        fx.inputs("Linear vector input", a -> layer.forward(a.get(0)), in(uniform(311, 4, -1f, 1f), 4));
        Tensor x = fx.constant(uniform(312, 4, -1f, 1f), 4);
        fx.parameters("Linear without bias", layer, () -> layer.forward(x));
    }

    @Test
    public void embedding() {
        Embedding layer = fx.keep(new Embedding(6, 3));
        layer.initialize(fx.backend(), new Random(INIT_SEED));
        Tensor indices = fx.int32(new int[]{0, 5, 5, 2, 0, 3}, 2, 3);
        fx.parameters("Embedding", layer, () -> layer.forward(indices));
    }

    /**
     * With a padding index the forward copies the padding row like any other, but the backward gives it no gradient
     * (PyTorch {@code padding_idx}); every other row gets the sum of the weights of the positions that select it.
     */
    @Test
    public void embeddingPaddingRowGetsNoGradient() {
        int rows = 4;
        int dim = 2;
        int padding = 2;
        int[] ids = {0, 2, 2, 1, 0};
        Embedding layer = fx.keep(new Embedding(rows, dim, padding));
        layer.initialize(fx.backend(), new Random(INIT_SEED));
        Tensor indices = fx.int32(ids, ids.length);
        int nodes = ComputationGraph.current().size();
        try (ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad();
             Tensor out = GradCheck.weighted(layer.forward(indices), Collections.emptyList(), 0)) {
            out.backward();
        }
        assertEquals(target + ": nodes left recorded", nodes, ComputationGraph.current().size());
        Tensor gradient = layer.weight().grad();
        assertNotNull(target + ": the table received no gradient", gradient);
        float[] w = GradCheck.weights(ids.length * dim, 0);
        float[] expected = new float[rows * dim];
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] != padding) {
                for (int d = 0; d < dim; d++) {
                    expected[ids[i] * dim + d] += w[i * dim + d];
                }
            }
        }
        assertArrayEquals(target + ": embedding gradient", expected, gradient.toFloatArray(), 1e-6f);
        layer.zeroGrad();
    }

    @Test
    public void conv2d() {
        Conv2d layer = fx.keep(new Conv2d(4, 6, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1},
                new int[]{1, 1}, 2, true));
        layer.initialize(fx.backend(), new Random(INIT_SEED));
        fx.write(layer.bias().tensor(), uniform(321, 6, -0.5f, 0.5f));
        fx.inputs("Conv2d input", a -> layer.forward(a.get(0)), in(uniform(322, 100, -1f, 1f), 1, 4, 5, 5));
        Tensor x = fx.constant(uniform(323, 200, -1f, 1f), 2, 4, 5, 5);
        fx.parameters("Conv2d grouped", layer, () -> layer.forward(x));

        Conv2d dilated = fx.keep(new Conv2d(2, 3, new int[]{2, 2}, new int[]{1, 1}, new int[]{1, 1},
                new int[]{2, 2}, 1, false));
        dilated.initialize(fx.backend(), new Random(INIT_SEED + 1));
        fx.inputs("Conv2d dilated input", a -> dilated.forward(a.get(0)), in(uniform(324, 50, -1f, 1f), 1, 2, 5, 5));
        Tensor y = fx.constant(uniform(325, 50, -1f, 1f), 1, 2, 5, 5);
        fx.parameters("Conv2d dilated", dilated, () -> dilated.forward(y));
    }

    @Test
    public void conv1d() {
        Conv1d layer = fx.keep(new Conv1d(3, 4, 3, 2, 1, 1, 1, true));
        layer.initialize(fx.backend(), new Random(INIT_SEED));
        fx.write(layer.bias().tensor(), uniform(331, 4, -0.5f, 0.5f));
        fx.inputs("Conv1d input", a -> layer.forward(a.get(0)), in(uniform(332, 42, -1f, 1f), 2, 3, 7));
        Tensor x = fx.constant(uniform(333, 42, -1f, 1f), 2, 3, 7);
        fx.parameters("Conv1d", layer, () -> layer.forward(x));

        Conv1d depthwise = fx.keep(new Conv1d(3, 3, 2, 1, 0, 2, 3, true));
        depthwise.initialize(fx.backend(), new Random(INIT_SEED + 1));
        fx.inputs("Conv1d depthwise input", a -> depthwise.forward(a.get(0)), in(uniform(334, 18, -1f, 1f), 1, 3, 6));
        Tensor y = fx.constant(uniform(335, 18, -1f, 1f), 1, 3, 6);
        fx.parameters("Conv1d depthwise", depthwise, () -> depthwise.forward(y));
    }

    @Test
    public void convTranspose2d() {
        ConvTranspose2d layer = fx.keep(new ConvTranspose2d(2, 4, new int[]{3, 3}, new int[]{2, 2},
                new int[]{1, 1}, new int[]{1, 0}, new int[]{1, 1}, 2, true));
        layer.initialize(fx.backend(), new Random(INIT_SEED));
        fx.write(layer.bias().tensor(), uniform(341, 4, -0.5f, 0.5f));
        fx.inputs("ConvTranspose2d input", a -> layer.forward(a.get(0)), in(uniform(342, 18, -1f, 1f), 1, 2, 3, 3));
        Tensor x = fx.constant(uniform(343, 36, -1f, 1f), 2, 2, 3, 3);
        fx.parameters("ConvTranspose2d", layer, () -> layer.forward(x));
    }

    @Test
    public void pooling() {
        MaxPool2d max = fx.keep(new MaxPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0},
                new int[]{1, 1}, true));
        MaxPool2d overlapping = fx.keep(new MaxPool2d(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1}));
        AvgPool2d avg = fx.keep(new AvgPool2d(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1}, false, false));
        AvgPool2d avgWithPad = fx.keep(new AvgPool2d(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1}, false,
                true));
        AdaptiveAvgPool2d adaptive = fx.keep(new AdaptiveAvgPool2d(3, 2));
        GradCheck.Input distinctValues = in(distinct(351, 50), 1, 2, 5, 5);
        GradCheck.Input smooth = in(uniform(352, 50, -1f, 1f), 1, 2, 5, 5);
        fx.inputs("MaxPool2d ceil mode", a -> max.forward(a.get(0)), distinctValues);
        fx.inputs("MaxPool2d overlapping", a -> overlapping.forward(a.get(0)), distinctValues);
        fx.inputs("AvgPool2d", a -> avg.forward(a.get(0)), smooth);
        fx.inputs("AvgPool2d counting padding", a -> avgWithPad.forward(a.get(0)), smooth);
        fx.inputs("AdaptiveAvgPool2d", a -> adaptive.forward(a.get(0)), in(uniform(353, 40, -1f, 1f), 1, 2, 5, 4));
    }

    @Test
    public void flatten() {
        Flatten all = fx.keep(new Flatten(1));
        Flatten middle = fx.keep(new Flatten(1, 2));
        GradCheck.Input x = in(uniform(361, 24, -1f, 1f), 2, 3, 2, 2);
        fx.inputs("Flatten from 1", a -> all.forward(a.get(0)), x);
        fx.inputs("Flatten 1..2", a -> middle.forward(a.get(0)), x);
    }

    @Test
    public void dropout() {
        Dropout layer = fx.keep(new Dropout(0.4f));
        GradCheck.Input x = in(uniform(371, 24, -1f, 1f), 4, 6);
        fx.inputs("Dropout training", a -> {
            layer.setSeed(77L);
            return layer.forward(a.get(0));
        }, x);
        layer.eval();
        fx.inputs("Dropout evaluation", a -> layer.forward(a.get(0)), x);
    }

    /** {@code Conv2d -> Tanh -> AvgPool2d -> Flatten -> Linear}: every parameter of a small smooth network. */
    @Test
    public void composedNetwork() {
        Conv2d conv = new Conv2d(2, 3, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
        Linear head = new Linear(12, 2);
        Sequential net = fx.keep(new Sequential(conv, new Tanh(),
                new AvgPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}), new Flatten(1), head));
        conv.initialize(fx.backend(), new Random(INIT_SEED));
        head.initialize(fx.backend(), new Random(INIT_SEED + 1));
        fx.write(conv.bias().tensor(), uniform(381, 3, -0.5f, 0.5f));
        fx.write(head.bias().tensor(), uniform(382, 2, -0.5f, 0.5f));
        fx.inputs("network input", a -> net.forward(a.get(0)), in(uniform(383, 32, -1f, 1f), 1, 2, 4, 4));
        Tensor x = fx.constant(uniform(384, 64, -1f, 1f), 2, 2, 4, 4);
        fx.parameters("network", net, () -> net.forward(x));
    }
}
