package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.*;

public class GradientCheckUtilTest {

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private boolean gpuAvailable = false;

    @Before
    public void setUp() {
        try {
            hwAccel = new HWAcceleration();

            if (hwAccel.getCLDevices().isEmpty()) {
                System.out.println("No OpenCL GPU devices available, skipping tests");
                return;
            }

            CLDevice clDevice = hwAccel.getCLDevices().get(0);
            backend = new OclComputeBackend(clDevice);
            gpuAvailable = true;
        } catch (Exception e) {
            System.out.println("OpenCL initialization failed: " + e.getMessage());
            gpuAvailable = false;
        }
    }

    @After
    public void tearDown() {
        if (backend != null) {
            try {
                backend.close();
            } catch (Exception e) {
                // Ignore close errors
            }
        }
        if (hwAccel != null) {
            try {
                hwAccel.shutdown();
            } catch (Exception e) {
                // Ignore shutdown errors
            }
        }
    }

    private void assumeGpuAvailable() {
        org.junit.Assume.assumeTrue("GPU not available", gpuAvailable);
    }

    @Test
    public void testGradientCheckUtilExists() {
        assertNotNull(GradientCheckUtil.class);
    }

    @Test
    public void testCheckAdd() {
        assumeGpuAvailable();
        assertTrue("Add gradient check should pass", GradientCheckUtil.checkAdd(backend));
    }

    @Test
    public void testCheckSub() {
        assumeGpuAvailable();
        assertTrue("Sub gradient check should pass", GradientCheckUtil.checkSub(backend));
    }

    @Test
    public void testCheckDiv() {
        assumeGpuAvailable();
        assertTrue("Div gradient check should pass", GradientCheckUtil.checkDiv(backend));
    }

    @Test
    public void testCheckMul() {
        assumeGpuAvailable();
        assertTrue("Mul gradient check should pass", GradientCheckUtil.checkMul(backend));
    }

    @Test
    public void testCheckMatmul() {
        assumeGpuAvailable();
        assertTrue("Matmul gradient check should pass", GradientCheckUtil.checkMatmul(backend));
    }

    @Test
    public void testCheckBmm() {
        assumeGpuAvailable();
        assertTrue("Bmm gradient check should pass", GradientCheckUtil.checkBmm(backend));
    }

    @Test
    public void testCheckExp() {
        assumeGpuAvailable();
        assertTrue("Exp gradient check should pass", GradientCheckUtil.checkExp(backend));
    }

    @Test
    public void testCheckLog() {
        assumeGpuAvailable();
        assertTrue("Log gradient check should pass", GradientCheckUtil.checkLog(backend));
    }

    @Test
    public void testCheckSqrt() {
        assumeGpuAvailable();
        assertTrue("Sqrt gradient check should pass", GradientCheckUtil.checkSqrt(backend));
    }

    @Test
    public void testCheckAbs() {
        assumeGpuAvailable();
        assertTrue("Abs gradient check should pass", GradientCheckUtil.checkAbs(backend));
    }

    @Test
    public void testCheckClamp() {
        assumeGpuAvailable();
        assertTrue("Clamp gradient check should pass", GradientCheckUtil.checkClamp(backend));
    }

    @Test
    public void testCheckNeg() {
        assumeGpuAvailable();
        assertTrue("Neg gradient check should pass", GradientCheckUtil.checkNeg(backend));
    }

    @Test
    public void testCheckPow() {
        assumeGpuAvailable();
        assertTrue("Pow gradient check should pass", GradientCheckUtil.checkPow(backend));
    }

    @Test
    public void testCheckSum() {
        assumeGpuAvailable();
        assertTrue("Sum gradient check should pass", GradientCheckUtil.checkSum(backend));
    }

    @Test
    public void testCheckMean() {
        assumeGpuAvailable();
        assertTrue("Mean gradient check should pass", GradientCheckUtil.checkMean(backend));
    }

    @Test
    public void testCheckMax() {
        assumeGpuAvailable();
        assertTrue("Max gradient check should pass", GradientCheckUtil.checkMax(backend));
    }

    @Test
    public void testCheckReshape() {
        assumeGpuAvailable();
        assertTrue("Reshape gradient check should pass", GradientCheckUtil.checkReshape(backend));
    }

    @Test
    public void testCheckTranspose() {
        assumeGpuAvailable();
        assertTrue("Transpose gradient check should pass", GradientCheckUtil.checkTranspose(backend));
    }

    @Test
    public void testCheckCat() {
        assumeGpuAvailable();
        assertTrue("Cat gradient check should pass", GradientCheckUtil.checkCat(backend));
    }

    @Test
    public void testCheckGather() {
        assumeGpuAvailable();
        assertTrue("Gather gradient check should pass", GradientCheckUtil.checkGather(backend));
    }

    @Test
    public void testCheckScatter() {
        assumeGpuAvailable();
        assertTrue("Scatter gradient check should pass", GradientCheckUtil.checkScatter(backend));
    }

    @Test
    public void testCheckGradientAccumulation() {
        assumeGpuAvailable();
        assertTrue("Gradient accumulation check should pass", GradientCheckUtil.checkGradientAccumulation(backend));
    }

    @Test
    public void testCheckNestedOperations() {
        assumeGpuAvailable();
        assertTrue("Nested operations check should pass", GradientCheckUtil.checkNestedOperations(backend));
    }

    @Test
    public void testGradCheckWithCustomTolerance() {
        assumeGpuAvailable();
        Tensor a = Tensor.rand(backend, Shape.of(2, 2)).withRequiresGrad(true);
        Tensor b = Tensor.rand(backend, Shape.of(2, 2)).withRequiresGrad(true);

        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheck(
                inputs -> TensorOps.add(inputs.get(0), inputs.get(1)),
                Arrays.asList(a, b),
                1e-3f, 5e-2f);

        assertTrue("Add gradient check with custom tolerance should pass", result.passed());

        a.close();
        b.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGradCheckNullFunction() {
        assumeGpuAvailable();
        Tensor a = Tensor.zeros(backend, Shape.of(2, 2));
        GradientCheckUtil.gradCheck(null, Arrays.asList(a));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGradCheckNullInputs() {
        GradientCheckUtil.gradCheck(inputs -> inputs.get(0), null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGradCheckEmptyInputs() {
        GradientCheckUtil.gradCheck(inputs -> inputs.get(0), Arrays.asList());
    }

    @Test
    public void testRunAllChecks() {
        assumeGpuAvailable();
        GradientCheckUtil.runAllChecks(backend);
    }

    // ------------------------------------------------------------------ parameter checks on the host reference path

    private static final float[] X = {0.5f, -0.25f, 0.75f, -0.5f, 0.25f, 0.125f};
    private static final float[] W = {0.3f, -0.6f, 0.9f, 0.2f, -0.4f, 0.7f};
    private static final float[] B = {0.1f, -0.2f};

    /**
     * {@code x @ weight + bias} through the recording facade, with a buffer {@code calls} that every forward pass
     * increments in place without recording (as running statistics are updated).
     */
    static final class CountingAffine extends Module {
        private final Parameter weight;
        private final Parameter bias;
        private final Tensor calls;

        CountingAffine(ComputeBackend b) {
            weight = registerParameter("weight", new Parameter(matrix(b, W, 3, 2), "weight"));
            bias = registerParameter("bias", new Parameter(Tensor.fromArray(b, B), "bias"));
            calls = registerBuffer("calls", Tensor.zeros(b, Shape.of(1)));
        }

        @Override
        public Tensor forward(Tensor input) {
            try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad();
                 Tensor one = Tensor.ones(calls.backend(), Shape.of(1))) {
                TensorOps.addInPlace(calls, one, 1.0f);
            }
            Tensor product = TensorOps.matmul(input, weight.tensor());
            try {
                return TensorOps.add(product, bias.tensor());
            } finally {
                closeIfUnrecorded(product);
            }
        }
    }

    /** {@code w * 1} recorded with a backward that doubles the gradient, so a gradient check must fail. */
    static final class WrongGradient extends Module {
        private final Parameter w;

        WrongGradient(ComputeBackend b) {
            w = registerParameter("w", new Parameter(Tensor.fromArray(b, B), "w"));
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor x = w.tensor();
            if (!Autograd.needsGrad(x)) {
                return TensorOps.mul(x, 1.0f);
            }
            Tensor raw = Autograd.untracked(() -> TensorOps.mul(x, 1.0f));
            return Autograd.record(raw, Collections.singletonList(x), out -> new GradientFunction() {
                @Override
                public List<Tensor> backward(Tensor gradOutput) {
                    return Collections.singletonList(TensorOps.mul(gradOutput, 2.0f));
                }

                @Override
                public List<Tensor> savedTensors() {
                    return Collections.emptyList();
                }
            });
        }
    }

    static Tensor matrix(ComputeBackend b, float[] values, int rows, int cols) {
        try (Tensor flat = Tensor.fromArray(b, values)) {
            return flat.reshape(rows, cols);
        }
    }

    /** Closes {@code t} unless it is the output of a recorded node (the backward pass or the check releases those). */
    static void closeIfUnrecorded(Tensor t) {
        if (ComputationGraph.current().getNode(t) == null) {
            t.close();
        }
    }

    /** {@code sum(model(x)^2)}, a new tensor per call; unrecorded intermediates are closed. */
    static Supplier<Tensor> squaredSum(Module model, Tensor x) {
        return () -> {
            Tensor y = model.forward(x);
            Tensor squared = null;
            try {
                squared = TensorOps.mul(y, y);
                return TensorOps.sum(squared);
            } finally {
                if (squared != null) {
                    closeIfUnrecorded(squared);
                }
                closeIfUnrecorded(y);
            }
        };
    }

    @Test
    public void testGradCheckParametersPassesAndRestoresState() {
        ComputeBackend host = TrainParity.hostBackend();
        CountingAffine model = new CountingAffine(host);
        try (Tensor x = matrix(host, X, 2, 3)) {
            Tensor stashed = matrix(host, new float[]{7f, 7f, 7f, 7f, 7f, 7f}, 3, 2);
            model.weight.setGrad(stashed);
            float[] callsBefore = model.calls.toFloatArray();
            int nodesBefore = ComputationGraph.current().size();
            boolean gradModeBefore = ComputationGraph.isGradEnabled();

            GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(
                    squaredSum(model, x), model, Arrays.asList(model.weight, model.bias), 1e-3f, 1e-2f, 100);

            assertTrue(result.toString(), result.passed());
            assertArrayEquals("weights restored", W, model.weight.tensor().toFloatArray(), 0f);
            assertArrayEquals("bias restored", B, model.bias.tensor().toFloatArray(), 0f);
            assertArrayEquals("buffer restored", callsBefore, model.calls.toFloatArray(), 0f);
            assertSame("stashed gradient back in its slot", stashed, model.weight.grad());
            assertArrayEquals(new float[]{7f, 7f, 7f, 7f, 7f, 7f}, stashed.toFloatArray(), 0f);
            assertNull("no analytical gradient left behind", model.bias.grad());
            assertEquals("nothing left recorded", nodesBefore, ComputationGraph.current().size());
            assertEquals(gradModeBefore, ComputationGraph.isGradEnabled());
        } finally {
            model.close();
            host.close();
        }
    }

    @Test
    public void testGradCheckParametersSamplesElements() {
        ComputeBackend host = TrainParity.hostBackend();
        CountingAffine model = new CountingAffine(host);
        try (Tensor x = matrix(host, X, 2, 3)) {
            GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(
                    squaredSum(model, x), null, Collections.singletonList(model.weight), 1e-3f, 1e-2f, 2);
            assertTrue(result.toString(), result.passed());
            assertNull("the checked slot is restored", model.weight.grad());
            assertNotNull("without the module, its other parameters keep the analytical gradient",
                    model.bias.grad());
        } finally {
            model.zeroGrad();
            model.close();
            host.close();
        }
    }

    @Test
    public void testGradCheckParametersDetectsWrongGradient() {
        ComputeBackend host = TrainParity.hostBackend();
        WrongGradient model = new WrongGradient(host);
        try {
            GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(
                    () -> model.forward(null), model, model.parameters(), 1e-3f, 1e-2f, 10);
            assertFalse("a doubled gradient must be reported", result.passed());
            assertTrue(result.message(), result.message().contains("'w'"));
            assertNull(model.w.grad());
        } finally {
            model.close();
            host.close();
        }
    }

    /**
     * A stashed gradient of a module parameter the check's backward pass does not reach stays tracked by the graph,
     * so {@link ComputationGraph#zeroGrad()} still clears it after the check.
     */
    @Test
    public void testGradCheckParametersKeepsUnreachedGradientTracked() {
        ComputeBackend host = TrainParity.hostBackend();
        CountingAffine model = new CountingAffine(host);
        try (Tensor x = matrix(host, X, 2, 3)) {
            try (Tensor first = squaredSum(model, x).get()) {
                first.backward();
            }
            Tensor weightGrad = model.weight.grad();
            assertNotNull(weightGrad);

            GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(
                    () -> TensorOps.mul(model.bias.tensor(), 3.0f), model, Collections.singletonList(model.bias),
                    1e-3f, 1e-2f, 10);

            assertTrue(result.toString(), result.passed());
            assertSame("the unreached slot holds its stashed gradient", weightGrad, model.weight.grad());
            ComputationGraph.current().zeroGrad();
            assertNull("the graph still clears the unreached parameter's gradient", model.weight.grad());
            assertNull(model.bias.grad());
        } finally {
            model.zeroGrad();
            model.close();
            host.close();
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGradCheckParametersNullLoss() {
        GradientCheckUtil.gradCheckParameters(null, null, Collections.emptyList(), 1e-3f, 1e-2f, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGradCheckParametersEmptyParameters() {
        GradientCheckUtil.gradCheckParameters(() -> null, null, Collections.emptyList(), 1e-3f, 1e-2f, 1);
    }

    @Test
    public void testGradCheckParametersRejectsFrozenParameter() {
        ComputeBackend host = TrainParity.hostBackend();
        CountingAffine model = new CountingAffine(host);
        try {
            model.bias.setRequiresGrad(false);
            assertThrows(IllegalArgumentException.class, () -> GradientCheckUtil.gradCheckParameters(
                    () -> null, model, Collections.singletonList(model.bias), 1e-3f, 1e-2f, 1));
            assertThrows(IllegalArgumentException.class, () -> GradientCheckUtil.gradCheckParameters(
                    () -> null, model, Collections.singletonList(model.weight), 0f, 1e-2f, 1));
        } finally {
            model.close();
            host.close();
        }
    }
}
