package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Test;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * The DDIM timestep schedule of {@link DDIMSampler}: the step whose next timestep is 0 lands on {@code abar_0}, and
 * only the step after the last timestep uses {@code abar_prev = 1} (diffusers' {@code final_alpha_cumprod}). With a
 * zero noise prediction and {@code eta = 0} each step scales the sample by {@code sqrt(abar_prev / abar_t)}, so the
 * intermediates expose the {@code abar_prev} every step used.
 */
public class DDIMSamplerScheduleTest {
    private static final long SEED = 77L;

    private final ComputeBackend backend = TrainParity.hostBackend();

    @After
    public void tearDown() {
        backend.close();
    }

    @Test
    public void theStepOntoTimestepZeroUsesItsAlphaCumprod() {
        LinearScheduler scheduler = new LinearScheduler(backend, 10);
        Shape shape = Shape.of(2, 3);
        float[] noise;
        try (Tensor draw = Tensor.randn(backend, shape, new Random(SEED).nextLong())) {
            noise = draw.toFloatArray();
        }
        DDIMSampler sampler = new DDIMSampler(new ZeroNoise(), scheduler, backend);
        DDIMSampler.SamplingResult result = sampler.sample(shape, 5, new Random(SEED));
        List<Tensor> steps = result.intermediates();
        try {
            int[] timesteps = {9, 7, 5, 3, 1, 0};
            assertEquals(timesteps.length, result.numSteps());
            assertEquals(timesteps.length, steps.size());
            double scale = 1.0;
            for (int i = 0; i < timesteps.length; i++) {
                double prev = i + 1 < timesteps.length ? scheduler.getAlphaCumprod(timesteps[i + 1]) : 1.0;
                scale *= Math.sqrt(prev / scheduler.getAlphaCumprod(timesteps[i]));
                float[] actual = steps.get(i).toFloatArray();
                for (int j = 0; j < noise.length; j++) {
                    assertEquals("step " + i + " element " + j, noise[j] * scale, actual[j],
                            1e-5 * Math.max(1.0, Math.abs(noise[j] * scale)));
                }
            }
            float[] samples = result.samples().toFloatArray();
            double end = 1.0 / Math.sqrt(scheduler.getAlphaCumprod(9));
            for (int j = 0; j < noise.length; j++) {
                assertEquals("telescoped sample", noise[j] * end, samples[j],
                        1e-5 * Math.max(1.0, Math.abs(noise[j] * end)));
            }
        } finally {
            for (Tensor t : steps) {
                t.close();
            }
            sampler.close();
        }
    }

    /** Predicts zero noise, so a DDIM step only rescales its input. */
    private static final class ZeroNoise extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return Tensor.zeros(input.backend(), input.shape());
        }
    }
}
