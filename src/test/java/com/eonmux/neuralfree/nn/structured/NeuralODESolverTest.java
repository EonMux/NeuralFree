package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * The fixed-step solvers of {@link NeuralODE} on the affine dynamics {@code f(z) = lambda * z + c} (one linear layer,
 * no activation), whose solution is {@code z(t) = (z0 + c / lambda) * e^(lambda t) - c / lambda}. A consistent
 * Runge-Kutta step advances a constant field by exactly {@code dt * c}, and the RK4 and Dormand-Prince steps reach
 * the closed form to within single-precision accumulation over the 49 steps.
 */
public class NeuralODESolverTest {
    private static final float LAMBDA = -0.7f;
    private static final float C = 0.3f;

    private final ComputeBackend backend = TrainParity.hostBackend();

    @After
    public void tearDown() {
        backend.close();
    }

    @Test
    public void constantFieldAdvancesByExactlyDtTimesTheField() {
        for (NeuralODE.Solver solver : NeuralODE.Solver.values()) {
            float[] out = integrate(solver, 0f, new float[]{1.5f, -2f});
            assertEquals(solver + " row 0", 1.5f + C, out[0], 5e-5f);
            assertEquals(solver + " row 1", -2f + C, out[1], 5e-5f);
        }
    }

    @Test
    public void higherOrderSolversMatchTheClosedForm() {
        float[] z0 = {1.5f, -2f};
        for (NeuralODE.Solver solver : new NeuralODE.Solver[]{NeuralODE.Solver.RK4, NeuralODE.Solver.DOPRI5}) {
            float[] out = integrate(solver, LAMBDA, z0);
            for (int i = 0; i < z0.length; i++) {
                double shift = C / (double) LAMBDA;
                double exact = (z0[i] + shift) * Math.exp(LAMBDA) - shift;
                assertEquals(solver + " row " + i, exact, out[i], 5e-5);
            }
        }
    }

    /** {@code z(1)} from {@code z(0) = z0} under {@code f(z) = lambda * z + c} with the given solver. */
    private float[] integrate(NeuralODE.Solver solver, float lambda, float[] z0) {
        try (NeuralODE ode = new NeuralODE(z0.length, 4, 1, solver, 1e-5f, 1e-8f)) {
            ode.initialize(backend, new Random(5));
            Linear layer = ode.layers().get(0);
            float[] weight = new float[z0.length * z0.length];
            float[] bias = new float[z0.length];
            for (int i = 0; i < z0.length; i++) {
                weight[i * z0.length + i] = lambda;
                bias[i] = C;
            }
            write(layer.weight().tensor(), weight);
            write(layer.bias().tensor(), bias);
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor flat = Tensor.fromArray(backend, z0);
                 Tensor input = flat.reshape(1, z0.length);
                 Tensor out = ode.forward(input)) {
                return out.toFloatArray();
            }
        }
    }

    private static void write(Tensor target, float[] values) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            GpuEvent done = target.storage().copyFromHostFloat(values, target.backend().transferQueue());
            if (done != null) {
                done.close();
            }
        }
    }
}
