package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.in;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.uniform;
import static org.junit.Assert.assertArrayEquals;

/**
 * Gradient checks of the normalization modules {@link BatchNorm1d}, {@link BatchNorm2d},
 * {@link GroupNorm}, {@link LayerNorm} and {@link RMSNorm}: the gradients a recorded backward pass gives the module's
 * input ({@link GradCheck#inputs}) and its affine parameters ({@link GradCheck#parameters}) against central finite
 * differences, on the host reference backend and on every available GPU API. The normalizations are smooth for inputs
 * with a positive variance, so no kink needs avoiding.
 *
 * <p>Inputs lie in {@code [-1, 1)}, and the affine parameters are overwritten with non-trivial
 * values (scales in {@code [0.5, 1)}, shifts in {@code [-0.5, 0.5)}). Batch normalization is checked in training
 * mode, where the batch statistics normalize the input and the gradient flows through them, and in evaluation mode
 * with non-trivial running statistics, which are constants. A training-mode forward updates the running statistics;
 * the parameter check restores them after every evaluation, and the test asserts that it left them as they
 * were.</p>
 */
@RunWith(Parameterized.class)
public class NormGradCheckTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return GradCheck.targets();
    }

    private final GradCheck.Target target;
    private GradCheck fx;

    public NormGradCheckTest(GradCheck.Target target) {
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
    public void batchNorm1dTraining() {
        BatchNorm1d bn = fx.keep(new BatchNorm1d(3));
        bn.initialize(fx.backend());
        affine(bn.weight().tensor(), bn.bias().tensor(), 501);
        fx.inputs("BatchNorm1d [B, F] input", a -> bn.forward(a.get(0)), in(uniform(502, 15, -1f, 1f), 5, 3));
        fx.inputs("BatchNorm1d [B, F, L] input", a -> bn.forward(a.get(0)), in(uniform(503, 24, -1f, 1f), 4, 3, 2));
        Tensor x = fx.constant(uniform(504, 24, -1f, 1f), 4, 3, 2);
        float[] mean = bn.runningMean().toFloatArray();
        float[] var = bn.runningVar().toFloatArray();
        fx.parameters("BatchNorm1d training", bn, () -> bn.forward(x));
        assertArrayEquals(target + ": running mean restored", mean, bn.runningMean().toFloatArray(), 0f);
        assertArrayEquals(target + ": running variance restored", var, bn.runningVar().toFloatArray(), 0f);
    }

    @Test
    public void batchNorm1dEvaluation() {
        BatchNorm1d bn = fx.keep(new BatchNorm1d(3));
        bn.initialize(fx.backend());
        affine(bn.weight().tensor(), bn.bias().tensor(), 511);
        runningStatistics(bn.runningMean(), bn.runningVar(), 512);
        bn.eval();
        fx.inputs("BatchNorm1d evaluation input", a -> bn.forward(a.get(0)), in(uniform(513, 12, -1f, 1f), 4, 3));
        Tensor x = fx.constant(uniform(514, 12, -1f, 1f), 4, 3);
        fx.parameters("BatchNorm1d evaluation", bn, () -> bn.forward(x));
    }

    @Test
    public void batchNorm2d() {
        BatchNorm2d bn = fx.keep(new BatchNorm2d(3));
        bn.initialize(fx.backend());
        affine(bn.weight().tensor(), bn.bias().tensor(), 521);
        GradCheck.Input input = in(uniform(522, 24, -1f, 1f), 2, 3, 2, 2);
        Tensor x = fx.constant(uniform(523, 24, -1f, 1f), 2, 3, 2, 2);
        fx.inputs("BatchNorm2d training input", a -> bn.forward(a.get(0)), input);
        fx.parameters("BatchNorm2d training", bn, () -> bn.forward(x));

        runningStatistics(bn.runningMean(), bn.runningVar(), 524);
        bn.eval();
        fx.inputs("BatchNorm2d evaluation input", a -> bn.forward(a.get(0)), input);
        fx.parameters("BatchNorm2d evaluation", bn, () -> bn.forward(x));
    }

    @Test
    public void groupNorm() {
        GroupNorm gn = fx.keep(new GroupNorm(2, 4));
        gn.initialize(fx.backend());
        affine(gn.weight().tensor(), gn.bias().tensor(), 531);
        fx.inputs("GroupNorm input", a -> gn.forward(a.get(0)), in(uniform(532, 24, -1f, 1f), 2, 4, 3));
        Tensor x = fx.constant(uniform(533, 32, -1f, 1f), 2, 4, 2, 2);
        fx.parameters("GroupNorm", gn, () -> gn.forward(x));

        GroupNorm plain = fx.keep(new GroupNorm(4, 4, 1e-5f, false));
        plain.initialize(fx.backend());
        fx.inputs("GroupNorm without affine input", a -> plain.forward(a.get(0)),
                in(uniform(534, 24, -1f, 1f), 2, 4, 3));
    }

    @Test
    public void layerNorm() {
        LayerNorm ln = fx.keep(new LayerNorm(new int[]{2, 3}));
        ln.initialize(fx.backend());
        affine(ln.weight().tensor(), ln.bias().tensor(), 541);
        fx.inputs("LayerNorm input", a -> ln.forward(a.get(0)), in(uniform(542, 24, -1f, 1f), 4, 2, 3));
        Tensor x = fx.constant(uniform(543, 18, -1f, 1f), 3, 2, 3);
        fx.parameters("LayerNorm", ln, () -> ln.forward(x));

        LayerNorm plain = fx.keep(new LayerNorm(4, 1e-5f, false));
        plain.initialize(fx.backend());
        fx.inputs("LayerNorm without affine input", a -> plain.forward(a.get(0)), in(uniform(544, 12, -1f, 1f), 3, 4));
    }

    @Test
    public void rmsNorm() {
        RMSNorm rms = fx.keep(new RMSNorm(5));
        rms.initialize(fx.backend());
        fx.write(rms.weight().tensor(), uniform(551, 5, 0.5f, 1f));
        fx.inputs("RMSNorm input", a -> rms.forward(a.get(0)), in(uniform(552, 15, -1f, 1f), 3, 5));
        Tensor x = fx.constant(uniform(553, 20, -1f, 1f), 2, 2, 5);
        fx.parameters("RMSNorm", rms, () -> rms.forward(x));

        RMSNorm plain = fx.keep(new RMSNorm(4, 1e-6f, false));
        plain.initialize(fx.backend());
        fx.inputs("RMSNorm without weight input", a -> plain.forward(a.get(0)), in(uniform(554, 8, -1f, 1f), 2, 4));
    }

    private void affine(Tensor weight, Tensor bias, long seed) {
        fx.write(weight, uniform(seed, Math.toIntExact(weight.numElements()), 0.5f, 1f));
        fx.write(bias, uniform(seed + 1, Math.toIntExact(bias.numElements()), -0.5f, 0.5f));
    }

    private void runningStatistics(Tensor mean, Tensor var, long seed) {
        fx.write(mean, uniform(seed, Math.toIntExact(mean.numElements()), -0.5f, 0.5f));
        fx.write(var, uniform(seed + 1, Math.toIntExact(var.numElements()), 0.5f, 1.5f));
    }
}
