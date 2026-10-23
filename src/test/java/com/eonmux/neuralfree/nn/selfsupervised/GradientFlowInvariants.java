package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.generative.GANTrainer;
import com.eonmux.neuralfree.nn.generative.SpectralNorm;
import com.eonmux.neuralfree.nn.generative.VAE;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.selfsupervised.SelfSupervisedGenerativeParityTest.LinearVAE;
import com.eonmux.neuralfree.nn.selfsupervised.SelfSupervisedGenerativeParityTest.Scenarios;
import com.eonmux.neuralfree.nn.selfsupervised.SelfSupervisedGenerativeParityTest.TokenwiseLinear;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The gradient-flow invariants {@link SelfSupervisedGenerativeParityTest} checks on every path: which parameters the
 * recorded graphs of the self-supervised and generative modules reach, what stays frozen or constant and
 * what records nothing. Each check runs in grad mode, inside the caller's step scope where it records.
 */
final class GradientFlowInvariants {
    private GradientFlowInvariants() {
    }

    static void clip(String what, ComputeBackend b) {
        try (CLIPTrainer clip = new CLIPTrainer(Scenarios.linear(b, 6, 5, 11), Scenarios.linear(b, 4, 3, 12),
                5, 3, 4);
             Tensor image = Scenarios.input(b, 14, -1f, 1f, 3, 6);
             Tensor text = Scenarios.input(b, 15, -1f, 1f, 3, 4)) {
            clip.setRandom(new Random(13));
            clip.initialize(b);
            assertEquals(what + " initial CLIP temperature", 0.07f, clip.temperature(), 1e-5f);
            CLIPTrainer.CLIPOutput out = clip.forwardCLIP(image, text);
            try (Tensor img = out.imageEmbedding(); Tensor txt = out.textEmbedding();
                 Tensor perImage = out.logitsPerImage(); Tensor perText = out.logitsPerText();
                 Tensor loss = out.clipLoss()) {
                loss.backward();
            }
            assertNotNull(what + " CLIP logit scale gradient", clip.getParameter("logit_scale").grad());
            assertNotNull(what + " CLIP image projection gradient",
                clip.getParameter("image_projection.weight").grad());
            assertNotNull(what + " CLIP text projection gradient",
                clip.getParameter("text_projection.weight").grad());
            assertNotNull(what + " CLIP image encoder gradient", clip.getParameter("image_encoder.weight").grad());
            assertNotNull(what + " CLIP text encoder gradient", clip.getParameter("text_encoder.weight").grad());
        }
    }

    static void dino(String what, ComputeBackend b) {
        Linear student = Scenarios.linear(b, 6, 8, 31);
        try (DINOTrainer dino = new DINOTrainer(student, 8);
             Tensor g1 = Scenarios.input(b, 33, -1f, 1f, 3, 6);
             Tensor g2 = Scenarios.input(b, 34, -1f, 1f, 3, 6)) {
            dino.initializeTeacher(Scenarios.linear(b, 6, 8, 32));
            Parameter teacherWeight = dino.teacher().getParameter("weight");
            assertFalse(what + " the DINO teacher is frozen", teacherWeight.requiresGrad());
            DINOTrainer.DINOOutput out = dino.forwardDINO(g1, g2);
            try (Tensor s = out.studentOutput(); Tensor t = out.teacherOutput(); Tensor loss = out.dinoLoss()) {
                assertFalse(what + " the DINO teacher output is a constant", t.requiresGrad());
                loss.backward();
            }
            assertNotNull(what + " DINO student gradient", student.getParameter("weight").grad());
            assertNull(what + " no DINO teacher gradient", teacherWeight.grad());
            dino.updateTeacher();
            assertFalse(what + " the DINO teacher stays frozen", teacherWeight.requiresGrad());
        }
    }

    static void mae(String what, ComputeBackend b) {
        try (MaskedAutoencoder mae = new MaskedAutoencoder(new TokenwiseLinear(b, 8, 8, 41),
                new TokenwiseLinear(b, 8, 8, 42), 2, 4, 6, 2, 0.5f);
             Tensor image = Scenarios.input(b, 44, -1f, 1f, 2, 2, 4, 6)) {
            mae.setRandom(new Random(43));
            MaskedAutoencoder.MAEOutput out = mae.forwardMAE(image);
            try (Tensor recon = out.reconstruction(); Tensor encoded = out.encoded();
                 Tensor decoded = out.decoded(); Tensor loss = out.maskedLoss()) {
                loss.backward();
            }
            assertEveryGradient(what + " MAE encoder", mae.encoder());
            assertEveryGradient(what + " MAE decoder", mae.decoder());
            try (Tensor recon = mae.forward(image)) {
                assertTrue(what + " the MAE forward reconstruction is recorded", recon.requiresGrad());
            }
        }
    }

    /** The real and fake passes of a spectrally normalized layer in one loss, each through its own sigma. */
    static void spectralNorm(String what, ComputeBackend b) {
        Linear wrapped = Scenarios.linear(b, 5, 3, 81);
        try (SpectralNorm sn = new SpectralNorm(wrapped, 1.0f, 3);
             Tensor real = Scenarios.input(b, 83, -1f, 1f, 2, 5);
             Tensor fake = Scenarios.input(b, 84, -1f, 1f, 2, 5)) {
            sn.initialize(b, new Random(82));
            try (Tensor yReal = sn.forward(real);
                 Tensor yFake = sn.forward(fake);
                 Tensor realSum = TensorOps.sum(yReal);
                 Tensor fakeSum = TensorOps.sum(yFake);
                 Tensor loss = TensorOps.sub(fakeSum, realSum)) {
                assertTrue(what + " the spectrally normalized forward is recorded", yReal.requiresGrad());
                loss.backward();
            }
            assertNotNull(what + " spectral norm weight gradient", wrapped.getParameter("weight").grad());
            assertNotNull(what + " spectral norm bias gradient", wrapped.getParameter("bias").grad());
            assertFalse(what + " the power-iteration buffers are constants",
                sn.namedBuffers().get("u").requiresGrad());
        }
    }

    static void vae(String what, ComputeBackend b) {
        try (LinearVAE vae = new LinearVAE(b, 4, 2);
             Tensor x = Scenarios.input(b, 75, 0.05f, 0.95f, 3, 4)) {
            vae.setRandom(new Random(74));
            try (Tensor recon = vae.forward(x);
                 Tensor kl = vae.takeAuxiliaryLoss()) {
                assertNotNull(what + " a training VAE forward sets the KL term", kl);
                assertTrue(what + " the KL term is recorded", kl.requiresGrad());
                try (Tensor diff = TensorOps.sub(recon, x);
                     Tensor squared = TensorOps.mul(diff, diff);
                     Tensor mse = TensorOps.mean(squared);
                     Tensor total = TensorOps.add(mse, kl)) {
                    total.backward();
                }
            }
            assertEveryGradient(what + " VAE (reconstruction + auxiliary KL)", vae);
            vae.zeroGrad();

            vae.eval();
            try (Tensor recon = vae.forward(x)) {
                assertNull(what + " no KL term in eval mode", vae.takeAuxiliaryLoss());
            }
            vae.train();

            VAE.VAEResult out = vae.forwardVAE(x);
            try (Tensor recon = out.reconstruction(); Tensor latent = out.latent();
                 Tensor mean = out.mean(); Tensor logVar = out.logVar();
                 VAE.VAELoss loss = vae.elboLoss(x, recon, mean, logVar)) {
                loss.totalLossTensor().backward();
            }
            assertEveryGradient(what + " VAE (ELBO total)", vae);
        }
    }

    static void augment(String what, ComputeBackend b) {
        try (ContrastiveLearner learner = new ContrastiveLearner(Scenarios.linear(b, 6, 5, 21), 5, 3, 0.5f);
             Tensor view = Scenarios.input(b, 23, -1f, 1f, 4, 6);
             Tensor leaf = view.withRequiresGrad(true)) {
            learner.setRandom(new Random(25));
            try (Tensor augmented = learner.augment(leaf, 0.1f, 0.25f)) {
                assertTrue(what + " augment records for an input that requires grad", augmented.requiresGrad());
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor augmented = learner.augment(leaf, 0.1f, 0.25f)) {
                assertFalse(what + " augment under noGrad does not require grad", augmented.requiresGrad());
            }
        }
    }

    /** Generation APIs run under noGrad: nothing is recorded and the training mode is restored. */
    static void inference(String what, ComputeBackend b) {
        int nodes = ComputationGraph.current().size();
        Linear generator = Scenarios.linear(b, 3, 4, 61);
        Linear discriminator = Scenarios.linear(b, 4, 1, 62);
        SGD genOptimizer = new SGD(generator.parameters(), 0.01f);
        SGD discOptimizer = new SGD(discriminator.parameters(), 0.01f);
        try (GANTrainer trainer = new GANTrainer(generator, discriminator, genOptimizer, discOptimizer,
                GANTrainer.GANLoss.WGAN);
             Tensor latent = Scenarios.input(b, 65, -1f, 1f, 2, 3);
             Tensor fake = trainer.generate(latent)) {
            assertFalse(what + " GAN generation is not recorded", fake.requiresGrad());
            assertTrue(what + " generation restores the training mode", generator.isTraining());
        } finally {
            genOptimizer.close();
            discOptimizer.close();
            generator.close();
            discriminator.close();
        }
        assertEquals(what + " inference records no node", nodes, ComputationGraph.current().size());
    }

    private static void assertEveryGradient(String what, Module module) {
        for (Map.Entry<String, Parameter> entry : module.namedParameters().entrySet()) {
            assertNotNull(what + ": gradient of " + entry.getKey(), entry.getValue().grad());
        }
    }
}

