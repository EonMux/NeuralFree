package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.activations.Tanh;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorValues;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The float64 gradient checks ({@link GradientCheckUtil#gradCheckFloat64},
 * {@link GradientCheckUtil#gradCheckParametersFloat64}): the FLOAT32 analytical gradient against central
 * differences of a FLOAT64 evaluation. They pass on correct gradients, report wrong ones, resolve gradients a
 * float32 finite difference cannot, leave the module, the reference and the graph as they found them, and refuse
 * inputs that break their contract by name.
 */
public class GradientCheckFloat64Test {

    private static final double EPSILON = 1e-3;
    private static final float TOLERANCE = 1e-2f;

    private final List<AutoCloseable> owned = new ArrayList<>();
    private ComputeBackend host;

    @Before
    public void setUp() {
        ComputationGraph.setGradEnabled(true);
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() throws Exception {
        for (int i = owned.size() - 1; i >= 0; i--) {
            owned.get(i).close();
        }
        owned.clear();
        host.close();
    }

    // ------------------------------------------------------------------ inputs

    @Test
    public void inputCheckPassesOnASmoothFunction() {
        GradientCheckUtil.TensorFunction fn = in -> {
            try (Tensor product = TensorOps.matmul(in.get(0), in.get(1))) {
                return TensorOps.tanh(product);
            }
        };
        int before = ComputationGraph.current().size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckFloat64(fn,
            Arrays.asList(uniform(host, 1, 3, 4), uniform(host, 2, 4, 2)), EPSILON, TOLERANCE);
        assertTrue(result.toString(), result.passed());
        assertEquals("the check leaves the graph as it was", before, ComputationGraph.current().size());
        assertTrue(ComputationGraph.isGradEnabled());
    }

    /**
     * {@code sum(x * x)} at {@code x ~ 300}: the loss is about 1.4e6, whose float spacing (0.125) divided by
     * {@code 2 epsilon} swamps the gradient {@code 2x}; in double the central difference is exact to rounding.
     */
    @Test
    public void float64ReferenceResolvesWhatFloat32Cannot() {
        float[] x = TrainParity.uniform(3, 16, 299f, 301f);
        GradientCheckUtil.TensorFunction square = in -> TensorOps.mul(in.get(0), in.get(0));
        Tensor input = keep(Tensor.fromArray(host, x));

        GradientCheckUtil.GradientCheckResult float32 = GradientCheckUtil.gradCheck(square,
            Collections.singletonList(input), (float) EPSILON, TOLERANCE);
        GradientCheckUtil.GradientCheckResult float64 = GradientCheckUtil.gradCheckFloat64(square,
            Collections.singletonList(input), EPSILON, TOLERANCE);

        assertFalse("float32 differences are dominated by the loss's float spacing: " + float32, float32.passed());
        assertTrue(float64.toString(), float64.passed());
        assertTrue(float64.toString(), float64.maxDifference() < 1e-3f);
    }

    @Test
    public void inputCheckReportsAWrongGradient() {
        GradientCheckUtil.TensorFunction doubled = in -> doubledGradientIdentity(in.get(0));
        GradientCheckUtil.TensorFunction reference = in -> TensorOps.mul(in.get(0), 1.0f);
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckFloat64(doubled, reference,
            Collections.singletonList(uniform(host, 4, 2, 3)), EPSILON, TOLERANCE);
        assertFalse(result.passed());
        assertEquals(1.0f, result.maxDifference(), 1e-4f);
    }

    @Test
    public void inputCheckRefusesBrokenArgumentsByName() {
        GradientCheckUtil.TensorFunction identity = in -> TensorOps.mul(in.get(0), 1.0f);
        List<Tensor> inputs = Collections.singletonList(uniform(host, 5, 2, 2));
        assertThrows(IllegalArgumentException.class,
            () -> GradientCheckUtil.gradCheckFloat64(identity, null, inputs, EPSILON, TOLERANCE));
        assertThrows(IllegalArgumentException.class,
            () -> GradientCheckUtil.gradCheckFloat64(identity, inputs, 0.0, TOLERANCE));
        assertThrows(IllegalArgumentException.class,
            () -> GradientCheckUtil.gradCheckFloat64(identity, inputs, EPSILON, -1f));
        Tensor wide = keep(inputs.get(0).to(DType.FLOAT64));
        IllegalArgumentException dtype = assertThrows(IllegalArgumentException.class,
            () -> GradientCheckUtil.gradCheckFloat64(identity, Collections.singletonList(wide), EPSILON,
                TOLERANCE));
        assertTrue(dtype.getMessage(), dtype.getMessage().contains("FLOAT64"));
        Tensor constant = uniform(host, 6, 2, 2);
        IllegalStateException narrow = assertThrows(IllegalStateException.class,
            () -> GradientCheckUtil.gradCheckFloat64(identity, in -> TensorOps.mul(constant, 1.0f), inputs,
                EPSILON, TOLERANCE));
        assertTrue(narrow.getMessage(), narrow.getMessage().contains("FLOAT32"));
    }

    // ------------------------------------------------------------------ parameters

    /** A training-mode Linear, BatchNorm1d, Tanh, Linear stack whose reference loaded its state. */
    @Test
    public void parameterCheckPassesAndLeavesBothModulesAsTheyWere() {
        Sequential model = keep(net(host, 7));
        Sequential reference = keep(reference(model, host));
        Tensor x = uniform(host, 8, 6, 3);
        Tensor probe = uniform(host, 9, 6, 2);
        Tensor x64 = keep(x.to(DType.FLOAT64));
        Tensor probe64 = keep(probe.to(DType.FLOAT64));
        Parameter firstWeight = model.namedParameters().get("0.weight");
        Tensor stashed = Tensor.ones(host, firstWeight.shape());
        firstWeight.setGrad(stashed);
        Map<String, float[]> modelState = values(model);
        Map<String, float[]> referenceState = values(reference);
        int before = ComputationGraph.current().size();

        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParametersFloat64(
            probeSum(model, x, probe), model, model.parameters(), probeSum(reference, x64, probe64), reference,
            EPSILON, TOLERANCE, 4);

        assertTrue(result.toString(), result.passed());
        assertEquals("graph size", before, ComputationGraph.current().size());
        assertSame("stashed gradient back in its slot", stashed, firstWeight.grad());
        assertNull(model.namedParameters().get("0.bias").grad());
        assertStateEquals(modelState, values(model));
        assertStateEquals(referenceState, values(reference));
    }

    @Test
    public void parameterCheckReportsAWrongGradient() {
        GradientCheckUtilTest.WrongGradient model = keep(new GradientCheckUtilTest.WrongGradient(host));
        GradientCheckUtilTest.WrongGradient reference = keep(new GradientCheckUtilTest.WrongGradient(host));
        reference.to(DType.FLOAT64);
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParametersFloat64(
            () -> model.forward(null), model, model.parameters(), () -> reference.forward(null), reference,
            EPSILON, TOLERANCE, 10);
        assertFalse(result.passed());
        assertTrue(result.message(), result.message().contains("'w'"));
    }

    @Test
    public void parameterCheckRefusesAReferenceThatDoesNotHoldTheModulesState() {
        Sequential model = keep(net(host, 10));
        Sequential other = keep(net(host, 11));
        other.to(DType.FLOAT64);
        Supplier<Tensor> loss = probeSum(model, uniform(host, 12, 6, 3), uniform(host, 13, 6, 2));
        IllegalArgumentException values = assertThrows(IllegalArgumentException.class,
            () -> GradientCheckUtil.gradCheckParametersFloat64(loss, model, model.parameters(), () -> null, other,
                EPSILON, TOLERANCE, 2));
        assertTrue(values.getMessage(), values.getMessage().contains("'0.weight'"));

        Sequential float32 = keep(net(host, 14));
        float32.loadStateDict(model.stateDict());
        IllegalArgumentException dtype = assertThrows(IllegalArgumentException.class,
            () -> GradientCheckUtil.gradCheckParametersFloat64(loss, model, model.parameters(), () -> null,
                float32, EPSILON, TOLERANCE, 2));
        assertTrue(dtype.getMessage(), dtype.getMessage().contains("FLOAT64"));

        Sequential reference = keep(reference(model, host));
        Parameter stranger = keep(new Parameter(uniform(host, 15, 2), "stranger"));
        IllegalArgumentException foreign = assertThrows(IllegalArgumentException.class,
            () -> GradientCheckUtil.gradCheckParametersFloat64(loss, model, Collections.singletonList(stranger),
                () -> null, reference, EPSILON, TOLERANCE, 2));
        assertTrue(foreign.getMessage(), foreign.getMessage().contains("'stranger'"));

        assertThrows(IllegalArgumentException.class, () -> GradientCheckUtil.gradCheckParametersFloat64(loss, null,
            model.parameters(), () -> null, reference, EPSILON, TOLERANCE, 2));
        assertThrows(IllegalArgumentException.class, () -> GradientCheckUtil.gradCheckParametersFloat64(loss, model,
            model.parameters(), () -> null, reference, Double.POSITIVE_INFINITY, TOLERANCE, 2));
        IllegalStateException narrow = assertThrows(IllegalStateException.class,
            () -> GradientCheckUtil.gradCheckParametersFloat64(loss, model, model.parameters(), loss, reference,
                EPSILON, TOLERANCE, 2));
        assertTrue(narrow.getMessage(), narrow.getMessage().contains("FLOAT32"));
    }

    /** The analytical side runs on the device; the reference evaluates in FLOAT64 on the same backend. */
    @Test
    public void parameterCheckOnOpenCl() {
        TrainParity.Session session = TrainParity.open(TrainParity.Api.OPENCL);
        ComputeBackend gpu = session.backend();
        try {
            Sequential model = net(gpu, 16);
            Sequential reference = reference(model, gpu);
            try (Tensor x = uniform(gpu, 17, 6, 3);
                 Tensor probe = uniform(gpu, 18, 6, 2);
                 Tensor x64 = x.to(DType.FLOAT64);
                 Tensor probe64 = probe.to(DType.FLOAT64)) {
                GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParametersFloat64(
                    probeSum(model, x, probe), model, model.parameters(), probeSum(reference, x64, probe64),
                    reference, EPSILON, TOLERANCE, 4);
                assertTrue(result.toString(), result.passed());
            } finally {
                model.close();
                reference.close();
            }
        } finally {
            TrainGpu.release(gpu);
            session.close();
        }
    }

    // ------------------------------------------------------------------ support

    /** {@code x} recorded with a backward that doubles the gradient. */
    private static Tensor doubledGradientIdentity(Tensor x) {
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

    /** A training-mode {@code Linear(3, 4), BatchNorm1d(4), Tanh, Linear(4, 2)} initialized from {@code seed}. */
    private static Sequential net(ComputeBackend backend, long seed) {
        Linear first = new Linear(3, 4);
        BatchNorm1d norm = new BatchNorm1d(4);
        Linear last = new Linear(4, 2);
        Random random = new Random(seed);
        first.initialize(backend, random);
        norm.initialize(backend);
        last.initialize(backend, random);
        Sequential net = new Sequential(first, norm, new Tanh(), last);
        net.train();
        return net;
    }

    /** A FLOAT64 module of {@link #net}'s architecture holding {@code model}'s state. */
    private static Sequential reference(Sequential model, ComputeBackend backend) {
        Sequential reference = net(backend, 0);
        reference.loadStateDict(model.stateDict(), true);
        reference.to(DType.FLOAT64);
        return reference;
    }

    /** {@code sum(module(x) * probe)} as a new tensor per call. */
    private static Supplier<Tensor> probeSum(Module module, Tensor x, Tensor probe) {
        return () -> {
            try (Tensor out = module.forward(x);
                 Tensor weighted = TensorOps.mul(out, probe)) {
                return TensorOps.sum(weighted);
            }
        };
    }

    /** Every parameter and buffer value of {@code module} by name, as floats. */
    private static Map<String, float[]> values(Module module) {
        Map<String, float[]> state = new java.util.LinkedHashMap<>();
        for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
            state.put(e.getKey(), TensorValues.toFloats(e.getValue().tensor()));
        }
        for (Map.Entry<String, Tensor> e : module.namedBuffers().entrySet()) {
            state.put("buffer " + e.getKey(), TensorValues.toFloats(e.getValue()));
        }
        return state;
    }

    private static void assertStateEquals(Map<String, float[]> expected, Map<String, float[]> actual) {
        assertEquals(expected.keySet(), actual.keySet());
        for (Map.Entry<String, float[]> e : expected.entrySet()) {
            assertArrayEquals(e.getKey(), e.getValue(), actual.get(e.getKey()), 0f);
        }
    }

    /** A FLOAT32 tensor of {@code dims} uniform in [-1, 1] from {@code seed}, closed after the test. */
    private Tensor uniform(ComputeBackend backend, long seed, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        try (Tensor flat = Tensor.fromArray(backend, TrainParity.uniform(seed, n, -1f, 1f))) {
            Tensor t = flat.reshape(Shape.of(dims));
            if (backend == host) {
                keep(t);
            }
            return t;
        }
    }

    private <T extends AutoCloseable> T keep(T resource) {
        owned.add(resource);
        return resource;
    }
}
