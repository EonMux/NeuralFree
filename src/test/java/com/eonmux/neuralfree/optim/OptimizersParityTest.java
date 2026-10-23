package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * SGD, AdamW, RMSprop, Adagrad and LAMB: the CPU path stores into the parameter's own elements (offset views
 * included), and on each GPU API the device path matches the CPU path within the optimizer-step tolerance class
 * ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}) for parameters and every state tensor.
 */
@RunWith(Enclosed.class)
public class OptimizersParityTest {

    private static final int ROWS = 4;
    private static final int COLS = 75;
    private static final int STEPS = 3;
    /** Chained floating-point operations per element over {@link #STEPS} elementwise steps. */
    private static final int ELEMENTWISE_OPS = 8 * STEPS;

    /** The optimizer configurations under test, covering every branch of each CPU loop. */
    enum Kind {
        SGD_PLAIN, SGD_MOMENTUM, SGD_NESTEROV, ADAMW, ADAMW_NO_DECAY, RMSPROP, RMSPROP_CENTERED_MOMENTUM,
        ADAGRAD, LAMB, LAMB_NO_BIAS_CORRECTION;

        Optimizer create(List<Parameter> params) {
            switch (this) {
                case SGD_PLAIN:
                    return new SGD(params, 0.05f, 0.0f, 0.01f, false);
                case SGD_MOMENTUM:
                    return new SGD(params, 0.05f, 0.9f, 0.01f, false);
                case SGD_NESTEROV:
                    return new SGD(params, 0.05f, 0.9f, 0.0f, true);
                case ADAMW:
                    return new AdamW(params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
                case ADAMW_NO_DECAY:
                    return new AdamW(params, 0.01f, 0.8f, 0.99f, 1e-6f, 0.0f);
                case RMSPROP:
                    return new RMSprop(params, 0.01f, 0.9f, 1e-8f, 0.01f, false, 0.0f);
                case RMSPROP_CENTERED_MOMENTUM:
                    return new RMSprop(params, 0.01f, 0.9f, 1e-8f, 0.0f, true, 0.9f);
                case ADAGRAD:
                    return new Adagrad(params, 0.1f, 1e-10f, 0.01f, 0.1f);
                case LAMB:
                    return new LAMB(params, 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f, true);
                case LAMB_NO_BIAS_CORRECTION:
                    return new LAMB(params, 0.01f, 0.9f, 0.999f, 1e-6f, 0.0f, false);
                default:
                    throw new IllegalStateException("unknown kind " + this);
            }
        }

        /** Reduction length of one step: the trust-ratio norms of LAMB, else the elementwise chain. */
        int reductionLength(int elements) {
            return this == LAMB || this == LAMB_NO_BIAS_CORRECTION ? elements * STEPS : ELEMENTWISE_OPS;
        }
    }

    /** A parameter that is rows 1 and 2 of a {@code [ROWS, COLS]} tensor, so it starts at a non-zero offset. */
    static final class OffsetParameter implements AutoCloseable {
        final Tensor flat;
        final Tensor base;
        final Tensor rows;
        final Parameter param;

        OffsetParameter(ComputeBackend backend, float[] baseData) {
            this.flat = Tensor.fromArray(backend, baseData);
            this.base = flat.reshape(Shape.of(ROWS, COLS));
            this.rows = base.slice(0, 1, 3);
            this.param = new Parameter(rows);
        }

        @Override
        public void close() {
            param.close();
            rows.close();
            base.close();
            flat.close();
        }
    }

    static float[] baseData() {
        return TrainParity.uniform(11, ROWS * COLS, -1f, 1f);
    }

    static float[] gradient(int step) {
        return TrainParity.uniform(100 + step, 2 * COLS, -1f, 1f);
    }

    /** Runs {@link #STEPS} steps of {@code opt} on {@code param} with the seeded gradients. */
    static void run(Optimizer opt, Parameter param, ComputeBackend backend, List<Tensor> owned) {
        for (int s = 0; s < STEPS; s++) {
            Tensor flat = Tensor.fromArray(backend, gradient(s));
            Tensor g = flat.reshape(param.shape());
            owned.add(flat);
            owned.add(g);
            param.setGrad(g);
            opt.step();
        }
    }

    /** The state tensors of a state dict ({@code <name>_<index>} entries), read back by key; closes the state dict. */
    static Map<String, float[]> stateArrays(Optimizer opt) {
        Map<String, float[]> out = new TreeMap<>();
        Map<String, Object> state = opt.stateDict();
        try {
            for (Map.Entry<String, Object> e : state.entrySet()) {
                if (e.getValue() instanceof Tensor) {
                    out.put(e.getKey(), ((Tensor) e.getValue()).toFloatArray());
                }
            }
        } finally {
            Optimizer.closeStateDict(state);
        }
        return out;
    }

    static float[] rows(float[] data, int from, int to) {
        float[] out = new float[(to - from) * COLS];
        System.arraycopy(data, from * COLS, out, 0, out.length);
        return out;
    }

    /**
     * Optimizer-step class between two float results that each meet it against the double reference:
     * {@code |gpu - cpu| <= 2 * (1e-5 * sqrt(k) * mag + k * FLT_MIN)} with {@code mag = |before| + |cpu| +
     * |cpu - before|}; NaN matches NaN and infinities match exactly.
     */
    static void assertStepClass(String what, float[] before, float[] cpu, float[] gpu, int k) {
        assertEquals(what + ": length", cpu.length, gpu.length);
        for (int i = 0; i < cpu.length; i++) {
            if (Float.isNaN(cpu[i]) || Float.isNaN(gpu[i]) || Float.isInfinite(cpu[i]) || Float.isInfinite(gpu[i])) {
                if (!(Float.isNaN(cpu[i]) && Float.isNaN(gpu[i])) && cpu[i] != gpu[i]) {
                    fail(what + ": element " + i + ": cpu " + cpu[i] + ", gpu " + gpu[i]);
                }
                continue;
            }
            double mag = Math.abs((double) before[i]) + Math.abs((double) cpu[i])
                    + Math.abs((double) cpu[i] - before[i]);
            double bound = 2.0 * (1e-5 * Math.sqrt(k) * mag + k * (double) TrainParity.FLT_MIN);
            if (!(Math.abs((double) gpu[i] - cpu[i]) <= bound)) {
                fail(what + ": element " + i + ": cpu " + cpu[i] + ", gpu " + gpu[i] + ", bound " + bound);
            }
        }
    }

    /** The CPU path (a backend without train kernels). */
    public static class HostPath {
        private ComputeBackend host;
        private final List<Tensor> owned = new ArrayList<>();

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            for (Tensor t : owned) {
                t.close();
            }
            owned.clear();
        }

        @Test
        public void everyOptimizerStoresIntoAnOffsetView() {
            float[] base = baseData();
            for (Kind kind : Kind.values()) {
                Tensor denseFlat = Tensor.fromArray(host, rows(base, 1, 3));
                Tensor denseRows = denseFlat.reshape(Shape.of(2, COLS));
                owned.add(denseFlat);
                owned.add(denseRows);
                Parameter dense = new Parameter(denseRows);
                try (OffsetParameter view = new OffsetParameter(host, base)) {
                    Optimizer onView = kind.create(Collections.singletonList(view.param));
                    Optimizer onDense = kind.create(Collections.singletonList(dense));
                    try {
                        run(onView, view.param, host, owned);
                        run(onDense, dense, host, owned);
                        float[] after = view.base.toFloatArray();
                        TrainParity.assertBitIdentical(kind + " row 0 untouched", rows(base, 0, 1), rows(after, 0, 1));
                        TrainParity.assertBitIdentical(kind + " row 3 untouched", rows(base, 3, 4), rows(after, 3, 4));
                        TrainParity.assertBitIdentical(kind + " parameter rows", dense.tensor().toFloatArray(),
                                rows(after, 1, 3));
                        Map<String, float[]> denseState = stateArrays(onDense);
                        Map<String, float[]> viewState = stateArrays(onView);
                        assertEquals(kind + " state keys", denseState.keySet(), viewState.keySet());
                        for (String key : denseState.keySet()) {
                            TrainParity.assertBitIdentical(kind + " " + key, denseState.get(key), viewState.get(key));
                        }
                    } finally {
                        onView.close();
                        onDense.close();
                        dense.close();
                    }
                }
            }
        }

        @Test
        public void stepWithoutGradientLeavesTheParameter() {
            float[] base = baseData();
            for (Kind kind : Kind.values()) {
                try (OffsetParameter view = new OffsetParameter(host, base)) {
                    Optimizer opt = kind.create(Collections.singletonList(view.param));
                    try {
                        opt.step();
                        TrainParity.assertBitIdentical(kind + " untouched", base, view.base.toFloatArray());
                    } finally {
                        opt.close();
                    }
                }
            }
        }
    }

    /** Device path against the CPU path on each GPU API with the train kernels available. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private ComputeBackend host;
        private final List<Tensor> owned = new ArrayList<>();

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            try {
                for (Tensor t : owned) {
                    t.close();
                }
                owned.clear();
            } finally {
                session.close();
            }
        }

        @Test
        public void everyOptimizerMatchesTheCpuPath() {
            float[] base = baseData();
            for (Kind kind : Kind.values()) {
                try (OffsetParameter cpuParam = new OffsetParameter(host, base);
                     OffsetParameter gpuParam = new OffsetParameter(gpu, base)) {
                    Optimizer cpuOpt = kind.create(Collections.singletonList(cpuParam.param));
                    Optimizer gpuOpt = kind.create(Collections.singletonList(gpuParam.param));
                    try {
                        Map<String, float[]> stateBefore = stateArrays(cpuOpt);
                        run(cpuOpt, cpuParam.param, host, owned);
                        run(gpuOpt, gpuParam.param, gpu, owned);
                        int k = kind.reductionLength(2 * COLS);
                        float[] cpuAfter = cpuParam.base.toFloatArray();
                        float[] gpuAfter = gpuParam.base.toFloatArray();
                        TrainParity.assertBitIdentical(api + " " + kind + " rows 0 and 3 untouched",
                                concat(rows(base, 0, 1), rows(base, 3, 4)),
                                concat(rows(gpuAfter, 0, 1), rows(gpuAfter, 3, 4)));
                        assertStepClass(api + " " + kind + " parameter", rows(base, 1, 3), rows(cpuAfter, 1, 3),
                                rows(gpuAfter, 1, 3), k);
                        Map<String, float[]> cpuState = stateArrays(cpuOpt);
                        Map<String, float[]> gpuState = stateArrays(gpuOpt);
                        assertEquals(api + " " + kind + " state keys", cpuState.keySet(), gpuState.keySet());
                        for (String key : cpuState.keySet()) {
                            assertStepClass(api + " " + kind + " " + key, stateBefore.get(key), cpuState.get(key),
                                    gpuState.get(key), k);
                        }
                    } finally {
                        cpuOpt.close();
                        gpuOpt.close();
                    }
                }
            }
        }

        private static float[] concat(float[] a, float[] b) {
            float[] out = new float[a.length + b.length];
            System.arraycopy(a, 0, out, 0, a.length);
            System.arraycopy(b, 0, out, a.length, b.length);
            return out;
        }
    }
}
