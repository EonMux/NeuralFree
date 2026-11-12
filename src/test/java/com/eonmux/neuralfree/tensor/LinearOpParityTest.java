package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;

/**
 * {@link TensorOps#linear} on each GPU API: the bias-prefilled GEMM gives the bits of the device's matmul and add, the
 * op records one node, its gradients (the weight gradient computed in {@code [out, in]}) have the bits of the
 * composition's on the device (see {@link LinearOpChecks} for the batched weight gradient), and a weight of no rows
 * gives zero gradients. Skipped where an API or the train kernels are unavailable.
 */
@RunWith(Parameterized.class)
public class LinearOpParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private LinearOpChecks checks;

    public LinearOpParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        session = TrainParity.open(api);
        checks = new LinearOpChecks(session.backend());
    }

    @After
    public void tearDown() {
        try {
            if (checks != null) {
                checks.close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    @Test
    public void matchesMatmulAndAddOnTheDevice() {
        checks.matchesTheComposition(new int[]{48}, true);
        checks.matchesTheComposition(new int[]{48}, false);
        checks.matchesTheComposition(new int[]{3, 16}, true);
        checks.matchesTheComposition(new int[]{}, true);
    }

    @Test
    public void anEmptyWeightGivesZeroGradientsOnTheDevice() {
        checks.anEmptyWeightGivesZeroGradients(true);
        checks.anEmptyWeightGivesZeroGradients(false);
    }
}
