package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.awayFrom;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.in;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.uniform;

/**
 * Gradient checks of the twelve activation modules: the gradient a recorded backward pass gives
 * the module's input against central finite differences ({@link GradCheck#inputs}), on the host reference backend
 * and on every available GPU API. The modules have no parameters.
 *
 * <p>Kinks: {@link ReLU}, {@link LeakyReLU} and {@link ELU} (whose derivative jumps at 0 for {@code alpha != 1}) read
 * inputs at least {@link GradCheck#KINK_MARGIN} from 0 ({@link GradCheck#awayFrom}); the other activations are smooth
 * and read inputs in {@code [-1, 1)}.</p>
 */
@RunWith(Parameterized.class)
public class ActivationsGradCheckTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return GradCheck.targets();
    }

    private final GradCheck.Target target;
    private GradCheck fx;

    public ActivationsGradCheckTest(GradCheck.Target target) {
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
    public void piecewiseActivations() {
        GradCheck.Input x = in(awayFrom(401, 24, -1f, 1f, 0f), 4, 6);
        check("ReLU", new ReLU(), x);
        check("LeakyReLU", new LeakyReLU(0.2f), x);
        check("ELU", new ELU(0.5f), x);
    }

    @Test
    public void smoothActivations() {
        GradCheck.Input x = in(uniform(411, 24, -1f, 1f), 4, 6);
        check("GELU", new GELU(), x);
        check("SiLU", new SiLU(), x);
        check("Sigmoid", new Sigmoid(), x);
        check("Tanh", new Tanh(), x);
        check("Mish", new Mish(), x);
    }

    @Test
    public void softmaxActivations() {
        GradCheck.Input x = in(uniform(421, 24, -1f, 1f), 2, 3, 4);
        check("Softmax last dim", new Softmax(), x);
        check("Softmax dim 1", new Softmax(1), x);
        check("LogSoftmax last dim", new LogSoftmax(), x);
        check("LogSoftmax dim 0", new LogSoftmax(0), x);
    }

    @Test
    public void gatedActivations() {
        GradCheck.Input x = in(uniform(431, 24, -1f, 1f), 2, 3, 4);
        check("GLU last dim", new GLU(), x);
        check("GeGLU last dim", new GeGLU(), x);
        GradCheck.Input y = in(uniform(432, 24, -1f, 1f), 2, 4, 3);
        check("GLU dim 1", new GLU(1), y);
        check("GeGLU dim 1", new GeGLU(1), y);
    }

    private void check(String what, Module activation, GradCheck.Input x) {
        Module module = fx.keep(activation);
        fx.inputs(what, a -> module.forward(a.get(0)), x);
    }
}
