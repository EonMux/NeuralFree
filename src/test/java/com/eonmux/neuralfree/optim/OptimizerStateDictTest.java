package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@link Optimizer#stateDict()} contract for all six optimizers: the caller owns a snapshot (new tensors that
 * later steps do not change, step count and hyper-parameters as boxed values) that round-trips through
 * {@link Optimizer#loadStateDict(Map)}, an optimizer without parameters returns a well-formed map without tensors,
 * closing the snapshot leaves the live buffer count where it was, and a map that does not fit is refused before
 * anything is written. Runs on a host backend that declines kernels (the CPU paths).
 */
public class OptimizerStateDictTest {
    private static final Shape MATRIX = Shape.of(2, 3);
    private static final Shape VECTOR = Shape.vector(4);

    /** Every optimizer, configured so that it keeps every state buffer family it has. */
    private static final Map<String, Function<List<Parameter>, Optimizer>> OPTIMIZERS = optimizers();

    private LiveBufferBackend backend;
    private final List<AutoCloseable> owned = new ArrayList<>();

    @Before
    public void setUp() {
        backend = new LiveBufferBackend();
        TrainGpu.forBackend(backend);
    }

    @After
    public void tearDown() throws Exception {
        for (int i = owned.size() - 1; i >= 0; i--) {
            owned.get(i).close();
        }
        owned.clear();
        TrainGpu.release(backend);
        backend.close();
    }

    private static Map<String, Function<List<Parameter>, Optimizer>> optimizers() {
        Map<String, Function<List<Parameter>, Optimizer>> all = new LinkedHashMap<>();
        all.put("SGD", p -> new SGD(p, 0.1f, 0.9f, 0.01f, true));
        all.put("Adam", p -> new Adam(p, 0.01f, 0.8f, 0.95f, 1e-6f, 0.01f));
        all.put("AdamW", p -> new AdamW(p, 0.01f, 0.8f, 0.95f, 1e-6f, 0.02f));
        all.put("RMSprop", p -> new RMSprop(p, 0.01f, 0.9f, 1e-6f, 0.01f, true, 0.5f));
        all.put("Adagrad", p -> new Adagrad(p, 0.1f, 1e-8f, 0.01f, 0.25f));
        all.put("LAMB", p -> new LAMB(p, 0.01f, 0.8f, 0.95f, 1e-6f, 0.01f, true));
        return all;
    }

    @Test
    public void everyOptimizerRoundTripsItsState() {
        for (Map.Entry<String, Function<List<Parameter>, Optimizer>> factory : OPTIMIZERS.entrySet()) {
            String name = factory.getKey();
            Optimizer source = optimizer(factory.getValue(), parameters(1));
            Optimizer target = optimizer(factory.getValue(), parameters(2));
            train(source, 2);

            Map<String, Object> saved = source.stateDict();
            try {
                assertEquals(name, 2, saved.get(Optimizer.STEP_COUNT_KEY));
                assertTrue(name + " keeps state tensors", tensorCount(saved) > 0);
                target.loadStateDict(saved);
            } finally {
                Optimizer.closeStateDict(saved);
            }

            assertEquals(name, 2, target.stepCount());
            assertStatesEqual(name, hostCopy(source), hostCopy(target));
        }
    }

    @Test
    public void theSnapshotIsACopyThatLaterStepsDoNotChange() {
        for (Map.Entry<String, Function<List<Parameter>, Optimizer>> factory : OPTIMIZERS.entrySet()) {
            String name = factory.getKey();
            Optimizer optimizer = optimizer(factory.getValue(), parameters(3));
            train(optimizer, 1);
            Map<String, Object> before = hostCopy(optimizer);
            String firstKey = firstTensorKey(before);
            Map<String, Object> snapshot = optimizer.stateDict();
            try {
                Tensor copy = (Tensor) snapshot.get(firstKey);
                Tensor live = optimizer.stateBuffers().values().iterator().next().get(0);
                assertNotSame(name, live.storage().gpuBuffer(), copy.storage().gpuBuffer());
                assertEquals(name, MATRIX, copy.shape());

                train(optimizer, 1);

                assertStatesEqual(name, before, toHost(snapshot));
                assertFalse(name + " state changed after a step", Arrays.equals((float[]) before.get(firstKey),
                        (float[]) hostCopy(optimizer).get(firstKey)));
            } finally {
                Optimizer.closeStateDict(snapshot);
            }
        }
    }

    @Test
    public void anOptimizerWithoutParametersReturnsAWellFormedMapWithoutTensors() {
        for (Map.Entry<String, Function<List<Parameter>, Optimizer>> factory : OPTIMIZERS.entrySet()) {
            String name = factory.getKey();
            Optimizer empty = optimizer(factory.getValue(), Collections.emptyList());
            Map<String, Object> state = empty.stateDict();

            assertEquals(name, 0, state.get(Optimizer.STEP_COUNT_KEY));
            assertEquals(name, 0, tensorCount(state));
            assertTrue(name + " records its hyper-parameters", state.containsKey("weight_decay"));
            assertFalse(name + " does not record the learning rate", state.containsKey("lr"));
            for (Object value : state.values()) {
                assertTrue(name + ": " + value, value instanceof Number || value instanceof Boolean);
            }

            Map<String, Object> restored = new HashMap<>(state);
            restored.put(Optimizer.STEP_COUNT_KEY, 7L);
            empty.loadStateDict(restored);
            assertEquals(name, 7, empty.stepCount());
        }
        Optimizer empty = optimizer(OPTIMIZERS.get("Adam"), Collections.emptyList());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> empty.loadStateDict(Collections.singletonMap("moment_0", new float[]{1f})));
        assertTrue(e.getMessage(), e.getMessage().contains("'moment_0'")
                && e.getMessage().contains("0 parameter(s)"));
    }

    @Test
    public void stateDictAndCloseLeaveTheLiveBufferCountUnchanged() {
        for (Map.Entry<String, Function<List<Parameter>, Optimizer>> factory : OPTIMIZERS.entrySet()) {
            String name = factory.getKey();
            Optimizer optimizer = optimizer(factory.getValue(), parameters(4));
            train(optimizer, 1);
            int before = backend.liveBuffers();

            Map<String, Object> state = optimizer.stateDict();
            assertEquals(name + " allocates one buffer per state tensor", before + tensorCount(state),
                    backend.liveBuffers());
            optimizer.loadStateDict(state);
            Optimizer.closeStateDict(state);

            assertEquals(name, before, backend.liveBuffers());
        }
    }

    @Test
    public void hostArraysAndStepCountFormsAreAccepted() {
        Optimizer adam = optimizer(OPTIMIZERS.get("Adam"), parameters(5));
        float[] moment = {1f, 2f, 3f, 4f, 5f, 6f};
        Map<String, Object> state = new HashMap<>();
        state.put("moment_0", moment);
        state.put(Optimizer.STEP_COUNT_KEY, 4.0f);
        state.put("beta1", 0.5f);
        adam.loadStateDict(state);

        assertEquals(4, adam.stepCount());
        assertArrayEquals(moment, (float[]) hostCopy(adam).get("moment_0"), 0f);
        assertEquals(0.8f, ((Adam) adam).beta1(), 0f);

        try (Tensor step = Tensor.scalar(backend, 9f)) {
            adam.loadStateDict(Collections.singletonMap(Optimizer.STEP_COUNT_KEY, step));
        }
        assertEquals(9, adam.stepCount());
    }

    @Test
    public void aMapThatDoesNotFitIsRefusedBeforeAnythingIsWritten() {
        Optimizer adam = optimizer(OPTIMIZERS.get("Adam"), parameters(6));
        train(adam, 1);
        Map<String, Object> before = hostCopy(adam);

        Map<String, Object> state = new LinkedHashMap<>();
        state.put(Optimizer.STEP_COUNT_KEY, 5);
        state.put("moment_0", new float[]{9f, 9f, 9f, 9f, 9f, 9f});
        try (Tensor wrongShape = Tensor.zeros(backend, Shape.of(3, 2))) {
            state.put("velocity_0", wrongShape);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> adam.loadStateDict(state));
            assertTrue(e.getMessage(), e.getMessage().contains("'velocity_0'")
                    && e.getMessage().contains(Shape.of(3, 2).toString())
                    && e.getMessage().contains(MATRIX.toString()));
        }
        assertEquals(1, adam.stepCount());
        assertStatesEqual("Adam", before, hostCopy(adam));

        state.put("velocity_0", new float[3]);
        assertTrue(message(adam, state).contains("holds 3 values"));
        state.remove("velocity_0");
        state.put("moment_2", new float[4]);
        assertTrue(message(adam, state).contains("2 parameter(s)"));
        state.remove("moment_2");
        state.put(Optimizer.STEP_COUNT_KEY, -1);
        assertTrue(message(adam, state).contains(Optimizer.STEP_COUNT_KEY));
        state.put(Optimizer.STEP_COUNT_KEY, 1.5f);
        assertTrue(message(adam, state).contains(Optimizer.STEP_COUNT_KEY));
        state.put(Optimizer.STEP_COUNT_KEY, new float[]{1f, 2f});
        assertTrue(message(adam, state).contains(Optimizer.STEP_COUNT_KEY));
        state.put(Optimizer.STEP_COUNT_KEY, 1);
        state.put("moment_0", "values");
        assertTrue(message(adam, state).contains("Tensor or a float[]"));
        assertStatesEqual("Adam", before, hostCopy(adam));
    }

    @Test
    public void aBufferTheHyperParametersDoNotUseIsRefused() {
        Optimizer plain = optimizer(p -> new RMSprop(p, 0.01f), parameters(7));
        assertTrue(message(plain, Collections.singletonMap("avg_grad_0", new float[6]))
                .contains("no 'avg_grad_' buffer for parameter 0"));
        Optimizer sgd = optimizer(p -> new SGD(p, 0.1f), parameters(8));
        assertTrue(message(sgd, Collections.singletonMap("velocity_1", new float[4]))
                .contains("'velocity_1'"));
        Map<String, Object> nullKey = new HashMap<>();
        nullKey.put(null, 1f);
        assertThrows(IllegalArgumentException.class, () -> sgd.loadStateDict(nullKey));
    }

    @Test
    public void closeStateDictClosesEveryTensorAndLeavesOtherValues() {
        Tensor a = Tensor.zeros(backend, VECTOR);
        Tensor b = Tensor.zeros(backend, VECTOR);
        int before = backend.liveBuffers();
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("a", a);
        state.put(Optimizer.STEP_COUNT_KEY, 3);
        state.put("b", b);

        Optimizer.closeStateDict(state);

        assertEquals(before - 2, backend.liveBuffers());
        assertEquals(3, state.get(Optimizer.STEP_COUNT_KEY));
        assertThrows(IllegalStateException.class, a::toFloatArray);
    }

    private String message(Optimizer optimizer, Map<String, ?> state) {
        return assertThrows(IllegalArgumentException.class, () -> optimizer.loadStateDict(state)).getMessage();
    }

    /** Runs {@code steps} steps with seeded gradients on every parameter. */
    private void train(Optimizer optimizer, int steps) {
        for (int s = 0; s < steps; s++) {
            int k = 0;
            for (Parameter param : optimizer.parameters()) {
                param.setGrad(tensor(values((int) param.numElements(), 100 * s + k++), param.shape()));
            }
            optimizer.step();
        }
    }

    private Optimizer optimizer(Function<List<Parameter>, Optimizer> factory, List<Parameter> params) {
        Optimizer optimizer = factory.apply(params);
        owned.add(optimizer::close);
        return optimizer;
    }

    /** A matrix parameter and a vector parameter with seeded values. */
    private List<Parameter> parameters(int seed) {
        List<Parameter> params = new ArrayList<>();
        params.add(parameter(MATRIX, seed, "w"));
        params.add(parameter(VECTOR, seed + 50, "b"));
        return params;
    }

    private Parameter parameter(Shape shape, int seed, String name) {
        Parameter param = new Parameter(tensor(values((int) shape.numElements(), seed), shape), name);
        owned.add(param::close);
        return param;
    }

    private Tensor tensor(float[] data, Shape shape) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    private static float[] values(int n, int seed) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.sin(seed * 31 + i * 7 + 1) * 0.5f;
        }
        return out;
    }

    /** The optimizer's state dict with tensors read to host arrays; the state dict is closed. */
    private static Map<String, Object> hostCopy(Optimizer optimizer) {
        Map<String, Object> state = optimizer.stateDict();
        try {
            return toHost(state);
        } finally {
            Optimizer.closeStateDict(state);
        }
    }

    private static Map<String, Object> toHost(Map<String, Object> state) {
        Map<String, Object> host = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : state.entrySet()) {
            Object v = e.getValue();
            host.put(e.getKey(), v instanceof Tensor ? ((Tensor) v).toFloatArray() : v);
        }
        return host;
    }

    private static void assertStatesEqual(String name, Map<String, Object> expected, Map<String, Object> actual) {
        assertEquals(name, expected.keySet(), actual.keySet());
        for (Map.Entry<String, Object> e : expected.entrySet()) {
            Object value = actual.get(e.getKey());
            if (e.getValue() instanceof float[]) {
                assertArrayEquals(name + " " + e.getKey(), (float[]) e.getValue(), (float[]) value, 0f);
            } else {
                assertEquals(name + " " + e.getKey(), e.getValue(), value);
            }
        }
    }

    private static int tensorCount(Map<String, Object> state) {
        int count = 0;
        for (Object value : state.values()) {
            if (value instanceof Tensor) {
                count++;
            }
        }
        return count;
    }

    private static String firstTensorKey(Map<String, Object> hostState) {
        for (Map.Entry<String, Object> e : hostState.entrySet()) {
            if (e.getValue() instanceof float[]) {
                return e.getKey();
            }
        }
        throw new AssertionError("no state tensor in " + hostState.keySet());
    }
}
