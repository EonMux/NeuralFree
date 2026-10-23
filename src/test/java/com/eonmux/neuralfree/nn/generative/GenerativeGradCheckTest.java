package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.activations.Tanh;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of the nn/generative modules with parameters, {@link VAE} (through a linear
 * subclass), {@link SpectralNorm} and {@link CFGWrapper} (whose parameters are those of the models it wraps), and of
 * the WGAN-GP discriminator loss of {@link GANTrainer}, on the CPU path and on each GPU API.
 *
 * <p>Each check compares the recorded gradient of a loss with central differences ({@link GradientCheckUtil}: step
 * {@value #EPSILON}, tolerance {@value #TOLERANCE}); inputs are drawn from {@code [-1, 1]} (from {@code [0.05, 0.95]}
 * for the VAE, whose binary cross-entropy reads them as targets). Randomness is fixed: the VAE's reparameterization
 * noise is drawn from a {@link Random} re-seeded in every evaluation, and the spectral norm's power-iteration vectors
 * are module buffers that the check restores after every evaluation.</p>
 *
 * <ul>
 *   <li><b>VAE.</b> The training forward with its {@code beta * KL} auxiliary term ({@link VAE#forward} plus
 *   {@link Module#takeAuxiliaryLoss()}), and the ELBO of {@link VAE#elboLoss} with both reconstruction losses. The
 *   decoder ends in a sigmoid, far inside the BCE clamp {@code [1e-7, 1 - 1e-7]}.</li>
 *   <li><b>Spectral norm.</b> One loss holds a discriminator pass over a real batch and one over a generated batch
 *   ({@code D(real) + D(G(z))}), each normalizing by its own {@code sigma}. The weight's singular values are about
 *   1.27, 0.70 and 0.41, so each forward's 30 training-mode power iterations converge (to {@code 0.3^30}) from the
 *   restored vectors to the singular vectors of the perturbed weight; {@code sigma} is then the top singular value,
 *   whose derivative {@code u v^T} is what the recorded graph gives with {@code u}, {@code v} held constant. In
 *   evaluation mode the stored vectors are used as they are, and {@code |u . W v|} is linear in {@code W}.</li>
 *   <li><b>WGAN-GP.</b> {@code mean(D(fake)) - mean(D(real)) + 10 * gp} with
 *   {@code gp = mean_b((sqrt(sum g_b^2 + 1e-8) - 1)^2)}, {@code g = d sum(D(x)) / dx} at a fixed interpolation
 *   {@code x}, formed exactly as {@link GANTrainer} forms it: the penalty's gradient reaches the discriminator through
 *   a create-graph pass ({@link Autograd#grad}), so this check passes only when every gradient function on the
 *   discriminator's path (linear maps and tanh) supports create-graph. The
 *   finite-difference evaluations run under {@code noGrad}, so the loss enables grad mode inside its own step to form
 *   {@code g} there and returns the value as a constant.</li>
 * </ul>
 */
@RunWith(Enclosed.class)
public class GenerativeGradCheckTest {

    static final float EPSILON = 1e-3f;
    static final float TOLERANCE = 1e-2f;
    static final int ELEMENTS = 6;
    static final float PENALTY_WEIGHT = 10.0f;

    /** A discriminator weight with singular values of about 1.27, 0.70 and 0.41. */
    static final float[] SPECTRAL_WEIGHT = {
        1.2f, 0.3f, -0.2f, 0.1f,
        0.2f, -0.5f, 0.1f, 0.3f,
        -0.1f, 0.2f, 0.4f, -0.3f};

    /** The checks, on the backend of the subclass. */
    public abstract static class Checks {

        protected abstract ComputeBackend backend();

        /** The training forward's reconstruction term plus its pending {@code beta * KL} term. */
        @Test
        public void vaeForwardWithKlTerm() {
            ComputeBackend b = backend();
            LinearVAE vae = new LinearVAE(b, 4, 2, 0.5f, 401);
            try (Tensor x = bounded(b, 402, 0.05f, 0.95f, 3, 4);
                 Tensor r = bounded(b, 403, -1.0f, 1.0f, 3, 4)) {
                vae.train();
                assertParameters("VAE forward + KL", () -> trainingLoss(vae, x, r), vae, vae.parameters(), ELEMENTS);
                assertInputs("VAE forward + KL input", vae.parameters(), in -> trainingLoss(vae, in.get(0), r), x);
            } finally {
                vae.close();
            }
        }

        @Test
        public void vaeElbo() {
            ComputeBackend b = backend();
            LinearVAE vae = new LinearVAE(b, 4, 2, 0.5f, 411);
            try (Tensor x = bounded(b, 412, 0.05f, 0.95f, 3, 4)) {
                for (String type : new String[]{"mse", "bce"}) {
                    assertParameters("VAE ELBO " + type, () -> elbo(vae, x, type), vae, vae.parameters(), ELEMENTS);
                }
            } finally {
                vae.close();
            }
        }

        /** {@code sum(tanh(D(real)) r1) + sum(tanh(D(G(z))) r2)} in the discriminator's and the generator's weights. */
        @Test
        public void spectralNormRealAndFake() {
            ComputeBackend b = backend();
            Linear inner = new Linear(4, 3);
            SpectralNorm discriminator = new SpectralNorm(inner, 1.0f, 30);
            Linear generator = linear(b, 2, 4, 421);
            try (Tensor real = bounded(b, 422, -1.0f, 1.0f, 5, 4);
                 Tensor z = bounded(b, 423, -1.0f, 1.0f, 5, 2);
                 Tensor r1 = bounded(b, 424, -1.0f, 1.0f, 5, 3);
                 Tensor r2 = bounded(b, 425, -1.0f, 1.0f, 5, 3)) {
                inner.initialize(b, new Random(426));
                Map<String, Object> state = new LinkedHashMap<>();
                state.put("weight", SPECTRAL_WEIGHT);
                state.put("bias", TrainParity.uniform(427, 3, -0.5f, 0.5f));
                inner.loadStateDict(state);
                discriminator.initialize(b, new Random(428));
                discriminator.train();
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                     Tensor converges = discriminator.forward(real)) {
                    assertTrue("finite discriminator output", Float.isFinite(converges.toFloatArray()[0]));
                }
                List<Parameter> params = new ArrayList<>(discriminator.parameters());
                params.addAll(generator.parameters());
                assertParameters("SpectralNorm D(real) + D(G(z))",
                    () -> realAndFake(discriminator, generator, real, z, r1, r2), discriminator, params, 12);

                discriminator.eval();
                assertParameters("SpectralNorm in evaluation mode",
                    () -> realAndFake(discriminator, generator, real, z, r1, r2), discriminator, params, 12);
            } finally {
                discriminator.close();
                generator.close();
            }
        }

        /** Both guidance paths of a wrapper over two distinct models. */
        @Test
        public void classifierFreeGuidance() {
            ComputeBackend b = backend();
            Linear conditional = linear(b, 3, 4, 431);
            Linear unconditional = linear(b, 3, 4, 432);
            CFGWrapper cfg = new CFGWrapper(conditional, unconditional, 2.0f);
            try (Tensor x = bounded(b, 433, -1.0f, 1.0f, 2, 3);
                 Tensor r = bounded(b, 434, -1.0f, 1.0f, 2, 4)) {
                List<Parameter> params = new ArrayList<>(conditional.parameters());
                params.addAll(unconditional.parameters());
                assertEquals("the wrapped models are registered children", params, cfg.parameters());
                assertParameters("CFGWrapper forward", () -> weighted(cfg.forward(x), r), cfg, params, ELEMENTS);
                assertParameters("CFGWrapper guidance 3.5", () -> weighted(cfg.forwardWithGuidance(x, 3.5f), r), cfg,
                    params, ELEMENTS);
                assertInputs("CFGWrapper input", params,
                    in -> weighted(cfg.forwardWithGuidance(in.get(0), 3.5f), r), x);
            } finally {
                cfg.close();
            }
        }

        /** The WGAN-GP discriminator loss (needs create-graph support on the discriminator's path). */
        @Test
        public void wganGradientPenalty() {
            ComputeBackend b = backend();
            Sequential discriminator = new Sequential(linear(b, 4, 5, 441), new Tanh(), linear(b, 5, 1, 442));
            Tensor interpolated = null;
            try (Tensor real = bounded(b, 443, -1.0f, 1.0f, 4, 4);
                 Tensor fake = bounded(b, 444, -1.0f, 1.0f, 4, 4);
                 Tensor mixed = interpolate(real, fake, new float[]{0.2f, 0.5f, 0.7f, 0.9f})) {
                interpolated = mixed.withRequiresGrad(true);
                Tensor x = interpolated;
                assertParameters("WGAN-GP discriminator loss", () -> wganGpLoss(discriminator, real, fake, x),
                    discriminator, discriminator.parameters(), ELEMENTS);
            } finally {
                if (interpolated != null) {
                    releaseLeaf(interpolated);
                }
                discriminator.close();
            }
        }
    }

    /** The CPU path. */
    public static class HostPath extends Checks {
        private ComputeBackend host;

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }

        @Override
        protected ComputeBackend backend() {
            return host;
        }
    }

    /** The device path on each GPU API. */
    @RunWith(Parameterized.class)
    public static class Device extends Checks {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
            session = TrainParity.open(api);
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            if (session != null) {
                session.close();
            }
        }

        @Override
        protected ComputeBackend backend() {
            return session.backend();
        }
    }

    /** A VAE with linear mean and log-variance heads and a linear decoder followed by a sigmoid. */
    static final class LinearVAE extends VAE {
        private final long seed;
        private final Linear meanHead;
        private final Linear logVarHead;
        private final Linear decoder;

        LinearVAE(ComputeBackend b, int inputDim, int latentDim, float beta, long seed) {
            super(latentDim, beta);
            this.seed = seed;
            this.meanHead = registerModule("meanHead", linear(b, inputDim, latentDim, seed + 1));
            this.logVarHead = registerModule("logVarHead", linear(b, inputDim, latentDim, seed + 2));
            this.decoder = registerModule("decoder", linear(b, latentDim, inputDim, seed + 3));
        }

        /** Re-seeds the reparameterization noise, so every evaluation draws the same noise. */
        void reseed() {
            setRandom(new Random(seed));
        }

        @Override
        protected EncodeResult encode(Tensor input) {
            Tensor mean = meanHead.forward(input);
            try {
                return new EncodeResult(mean, logVarHead.forward(input));
            } catch (RuntimeException e) {
                mean.close();
                throw e;
            }
        }

        @Override
        protected Tensor decode(Tensor latent) {
            try (Tensor logits = decoder.forward(latent)) {
                return TensorOps.sigmoid(logits);
            }
        }
    }

    /** {@code sum(forward(x) * r) + takeAuxiliaryLoss()} in training mode. */
    static Tensor trainingLoss(LinearVAE vae, Tensor x, Tensor r) {
        vae.reseed();
        Tensor reconstruction = vae.forward(x);
        Tensor kl = vae.takeAuxiliaryLoss();
        try (Tensor w = weighted(reconstruction, r);
             Tensor total = TensorOps.sum(w);
             Tensor klTerm = kl) {
            assertNotNull("a training forward with beta > 0 leaves its KL term pending", klTerm);
            try (Tensor scalarKl = klTerm.reshape(total.shape())) {
                return TensorOps.add(total, scalarKl);
            }
        }
    }

    /** The recorded ELBO total of {@link VAE#elboLoss} with reconstruction loss {@code type}, as a new tensor. */
    static Tensor elbo(LinearVAE vae, Tensor x, String type) {
        vae.reseed();
        VAE.VAEResult result = vae.forwardVAE(x);
        try (Tensor reconstruction = result.reconstruction();
             Tensor latent = result.latent();
             Tensor mean = result.mean();
             Tensor logVar = result.logVar();
             VAE.VAELoss loss = vae.elboLoss(x, reconstruction, mean, logVar, type)) {
            return TensorOps.add(loss.totalLossTensor(), 0.0f);
        }
    }

    /** {@code sum(tanh(D(real)) * r1) + sum(tanh(D(G(z))) * r2)}. */
    static Tensor realAndFake(Module discriminator, Module generator, Tensor real, Tensor z, Tensor r1, Tensor r2) {
        try (Tensor realOut = discriminator.forward(real);
             Tensor realSquashed = TensorOps.tanh(realOut);
             Tensor realTerms = TensorOps.mul(realSquashed, r1);
             Tensor realSum = TensorOps.sum(realTerms);
             Tensor generated = generator.forward(z);
             Tensor fakeOut = discriminator.forward(generated);
             Tensor fakeSquashed = TensorOps.tanh(fakeOut);
             Tensor fakeTerms = TensorOps.mul(fakeSquashed, r2);
             Tensor fakeSum = TensorOps.sum(fakeTerms)) {
            return TensorOps.add(realSum, fakeSum);
        }
    }

    /**
     * The WGAN-GP discriminator loss at the interpolation leaf {@code x}: recorded in grad mode; under
     * {@code noGrad} (the finite-difference evaluations) formed in grad mode inside its own step and returned as a
     * constant holding its value.
     */
    static Tensor wganGpLoss(Module discriminator, Tensor real, Tensor fake, Tensor x) {
        if (ComputationGraph.isGradEnabled()) {
            return recordedWganGpLoss(discriminator, real, fake, x);
        }
        float value;
        try (ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad();
             ComputationGraph.StepScope step = ComputationGraph.current().step();
             Tensor loss = recordedWganGpLoss(discriminator, real, fake, x)) {
            value = loss.toFloatArray()[0];
        }
        return Tensor.fromArray(real.backend(), new float[]{value});
    }

    /** {@code mean(D(fake)) + mean(-D(real)) + PENALTY_WEIGHT * gp(x)}, recorded. */
    static Tensor recordedWganGpLoss(Module discriminator, Tensor real, Tensor fake, Tensor x) {
        try (Tensor realOut = discriminator.forward(real);
             Tensor fakeOut = discriminator.forward(fake);
             Tensor negReal = TensorOps.neg(realOut);
             Tensor fakeLoss = TensorOps.mean(fakeOut);
             Tensor realLoss = TensorOps.mean(negReal);
             Tensor wgan = TensorOps.add(fakeLoss, realLoss);
             Tensor penalty = gradientPenalty(discriminator, x);
             Tensor weightedPenalty = TensorOps.mul(penalty, PENALTY_WEIGHT)) {
            return TensorOps.add(wgan, weightedPenalty);
        }
    }

    /**
     * {@code mean_b((sqrt(sum g_b^2 + 1e-8) - 1)^2)} with {@code g = d sum(D(x)) / dx} from a create-graph pass, as
     * {@link GANTrainer} forms it.
     */
    static Tensor gradientPenalty(Module discriminator, Tensor x) {
        Tensor gradient;
        try (Tensor output = discriminator.forward(x);
             Tensor total = TensorOps.sum(output)) {
            gradient = Autograd.grad(total, Collections.singletonList(x), null, true, true).get(0);
        }
        assertNotNull("the discriminator output depends on its input", gradient);
        int batch = x.shape().dim(0);
        int perSample = (int) (x.numElements() / batch);
        try (Tensor g = gradient;
             Tensor rows = g.reshape(batch, perSample);
             Tensor squares = TensorOps.mul(rows, rows);
             Tensor sumSquares = TensorOps.sum(squares, 1);
             Tensor shifted = TensorOps.add(sumSquares, 1e-8f);
             Tensor norm = TensorOps.sqrt(shifted);
             Tensor deviation = TensorOps.add(norm, -1.0f);
             Tensor squaredDeviation = TensorOps.mul(deviation, deviation)) {
            return TensorOps.mean(squaredDeviation);
        }
    }

    /** {@code alpha_b * real_b + (1 - alpha_b) * fake_b} per row, computed on the host, as a new constant tensor. */
    static Tensor interpolate(Tensor real, Tensor fake, float[] alpha) {
        float[] a = real.toFloatArray();
        float[] f = fake.toFloatArray();
        int perRow = a.length / alpha.length;
        float[] mixed = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            float w = alpha[i / perRow];
            mixed[i] = w * a[i] + (1.0f - w) * f[i];
        }
        try (Tensor flat = Tensor.fromArray(real.backend(), mixed)) {
            return flat.reshape(real.shape());
        }
    }

    /** Clears and closes the gradient the analytical pass accumulated into {@code leaf}, then closes the leaf. */
    static void releaseLeaf(Tensor leaf) {
        Tensor grad = leaf.grad();
        if (grad != null) {
            leaf.setGrad(null);
            grad.close();
        }
        leaf.close();
    }

    /** A {@link Linear} on {@code b} with seeded weights in {@code [-0.5, 0.5]}. */
    static Linear linear(ComputeBackend b, int in, int out, long seed) {
        Linear layer = new Linear(in, out);
        layer.initialize(b, new Random(seed));
        Map<String, Object> state = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : layer.stateDict().entrySet()) {
            int n = ((float[]) e.getValue()).length;
            state.put(e.getKey(), TrainParity.uniform(seed * 31 + e.getKey().hashCode(), n, -0.5f, 0.5f));
        }
        layer.loadStateDict(state);
        return layer;
    }

    /**
     * Asserts that the recorded gradients of {@code loss} in {@code params} match central differences on up to
     * {@code elements} elements per parameter, with {@code module}'s buffers restored after every evaluation, and
     * that the check leaves the graph unchanged.
     */
    static void assertParameters(String what, Supplier<Tensor> loss, Module module, List<Parameter> params,
                                 int elements) {
        assertFalse(what + " has no parameters", params.isEmpty());
        int before = ComputationGraph.current().size();
        GradientCheckUtil.GradientCheckResult result =
            GradientCheckUtil.gradCheckParameters(loss, module, params, EPSILON, TOLERANCE, elements);
        assertTrue(what + " parameter gradients: " + result, result.passed());
        assertEquals(what + ": the check leaves the graph as it found it", before, ComputationGraph.current().size());
    }

    /**
     * Asserts that the recorded gradient of {@code sum(fn(inputs))} in every input matches central differences; the
     * gradients the check leaves in {@code params} are cleared.
     */
    static void assertInputs(String what, List<Parameter> params, GradientCheckUtil.TensorFunction fn,
                             Tensor... inputs) {
        int before = ComputationGraph.current().size();
        GradientCheckUtil.GradientCheckResult result =
            GradientCheckUtil.gradCheck(fn, Arrays.asList(inputs), EPSILON, TOLERANCE);
        for (Parameter p : params) {
            p.zeroGrad();
        }
        assertTrue(what + " gradients: " + result, result.passed());
        assertEquals(what + ": the check leaves the graph as it found it", before, ComputationGraph.current().size());
    }

    /** {@code out * r} as a new tensor; {@code out} is closed. */
    static Tensor weighted(Tensor out, Tensor r) {
        try (Tensor o = out) {
            return TensorOps.mul(o, r);
        }
    }

    /** A FLOAT32 tensor of {@code dims} with seeded values in {@code [lo, hi]}. */
    static Tensor bounded(ComputeBackend b, long seed, float lo, float hi, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        try (Tensor flat = Tensor.fromArray(b, TrainParity.uniform(seed, n, lo, hi))) {
            return flat.reshape(Shape.of(dims));
        }
    }
}
