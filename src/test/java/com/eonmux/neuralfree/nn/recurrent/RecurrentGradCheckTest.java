package com.eonmux.neuralfree.nn.recurrent;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of the recurrent modules: for every module class with
 * parameters, the gradients of {@code sum(output * probe)} with respect to the module's inputs
 * ({@link GradientCheckUtil#gradCheck}) and to all of its parameters ({@link GradientCheckUtil#gradCheckParameters})
 * match central differences with step 1e-3 within 1e-2, on the host reference backend and on every available GPU API
 * ({@link TrainParity}). The probe is a fixed tensor in [-1, 1], so every output element carries its own weight.
 *
 * <p><b>Inputs and determinism.</b> Inputs, initial states and probes are uniform in [-1, 1] from fixed seeds; the
 * modules are built from a seeded {@link Random} and every bias gets non-zero seeded values. Dropout is 0 and the
 * modules run in evaluation mode, so every evaluation computes the same function.</p>
 *
 * <p><b>Kinks.</b> The cells use tanh and sigmoid only (the RNN runs its tanh activation), so the recurrences are
 * smooth. The state-space model's {@code a = exp(a_log)} starts near 0.5, far from the {@code |a| < 1e-8} branch of
 * its discretization, and its truncated and zero-padded state inputs are fixed selections.</p>
 *
 * <p><b>Lifetime.</b> Input checks run with the module frozen, so no parameter slot is written; parameter checks
 * restore every slot. Each check leaves the thread's graph at the size it had before the check.</p>
 */
@RunWith(Parameterized.class)
public class RecurrentGradCheckTest {

    private static final long SEED = 9090L;
    private static final float EPSILON = 1e-3f;
    private static final float TOLERANCE = 1e-2f;
    private static final int MAX_ELEMENTS = 6;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> backends() {
        List<Object[]> rows = new ArrayList<>();
        rows.add(new Object[]{"host", null});
        for (TrainParity.Api api : TrainParity.Api.values()) {
            rows.add(new Object[]{api.name(), api});
        }
        return rows;
    }

    private final String name;
    private final TrainParity.Api api;
    private final List<AutoCloseable> owned = new ArrayList<>();
    private TrainParity.Session session;
    private ComputeBackend backend;

    public RecurrentGradCheckTest(String name, TrainParity.Api api) {
        this.name = name;
        this.api = api;
    }

    @Before
    public void setUp() {
        ComputationGraph.setGradEnabled(true);
        if (api == null) {
            backend = TrainParity.hostBackend();
        } else {
            session = TrainParity.open(api);
            backend = session.backend();
        }
    }

    @After
    public void tearDown() throws Exception {
        Exception failure = null;
        for (int i = owned.size() - 1; i >= 0; i--) {
            try {
                owned.get(i).close();
            } catch (Exception e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        owned.clear();
        if (session != null) {
            session.close();
        } else if (backend != null) {
            TrainGpu.release(backend);
        }
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ cells

    @Test
    public void rnnCell() {
        RNNCell cell = keep(new RNNCell(3, 4));
        prepare(cell, (b, r) -> cell.initialize(b, r));
        Tensor x = tensor(11, 2, 3);
        Tensor h = tensor(12, 2, 4);
        Tensor probe = tensor(13, 2, 4);
        checkInputs("RNNCell", cell, in -> probeSum(cell.forward(in.get(0), in.get(1)), probe), x, h);
        checkParameters("RNNCell", cell, () -> probeSum(cell.forward(x, h), probe));
    }

    @Test
    public void gruCell() {
        GRUCell cell = keep(new GRUCell(3, 4));
        prepare(cell, (b, r) -> cell.initialize(b, r));
        Tensor x = tensor(21, 2, 3);
        Tensor h = tensor(22, 2, 4);
        Tensor probe = tensor(23, 2, 4);
        checkInputs("GRUCell", cell, in -> probeSum(cell.forward(in.get(0), in.get(1)), probe), x, h);
        checkParameters("GRUCell", cell, () -> probeSum(cell.forward(x, h), probe));
    }

    /** Both the new hidden state and the new cell state enter the checked sum, each with its own probe. */
    @Test
    public void lstmCell() {
        LSTMCell cell = keep(new LSTMCell(3, 4));
        prepare(cell, (b, r) -> cell.initialize(b, r));
        Tensor x = tensor(31, 2, 3);
        Tensor h = tensor(32, 2, 4);
        Tensor c = tensor(33, 2, 4);
        Tensor hiddenProbe = tensor(34, 2, 4);
        Tensor cellProbe = tensor(35, 2, 4);
        checkInputs("LSTMCell", cell,
            in -> lstmCellLoss(cell, in.get(0), in.get(1), in.get(2), hiddenProbe, cellProbe), x, h, c);
        checkParameters("LSTMCell", cell, () -> lstmCellLoss(cell, x, h, c, hiddenProbe, cellProbe));
    }

    // ------------------------------------------------------------------ sequence modules

    @Test
    public void stackedBidirectionalRnn() {
        RNN rnn = keep(new RNN(3, 4, 2, true, 0.0, RNNCell.TANH, true));
        prepare(rnn, (b, r) -> rnn.initialize(b, r));
        checkSequenceModule("RNN", rnn, 8);
    }

    @Test
    public void stackedBidirectionalGru() {
        GRU gru = keep(new GRU(3, 4, 2, true, 0.0f, true));
        prepare(gru, (b, r) -> gru.initialize(b, r));
        checkSequenceModule("GRU", gru, 8);
    }

    @Test
    public void stackedBidirectionalLstm() {
        LSTM lstm = keep(new LSTM(3, 4, 2, true, 0.0f, true));
        prepare(lstm, (b, r) -> lstm.initialize(b, r));
        checkSequenceModule("LSTM", lstm, 8);
    }

    /**
     * Two layers with state size 5: the first layer zero-pads its 3 input features to the state size, the second
     * truncates its 6 hidden features to it, so both state-input paths are checked.
     */
    @Test
    public void stackedStateSpaceModel() {
        StateSpaceModel ssm = keep(new StateSpaceModel(3, 6, 5, 2, true, 0.0f));
        prepare(ssm, (b, r) -> ssm.initialize(b, r));
        checkSequenceModule("StateSpaceModel", ssm, 6);
    }

    /** Input {@code [2, 4, 3]}, output {@code [2, 4, outFeatures]}. */
    private void checkSequenceModule(String what, Module module, int outFeatures) {
        Tensor x = tensor(41, 2, 4, 3);
        Tensor probe = tensor(42, 2, 4, outFeatures);
        checkInputs(what, module, in -> probeSum(module.forward(in.get(0)), probe), x);
        checkParameters(what, module, () -> probeSum(module.forward(x), probe));
    }

    // ------------------------------------------------------------------ checks

    /** The input gradients of {@code fn} at {@code inputs}, with {@code module}'s parameters frozen meanwhile. */
    private void checkInputs(String what, Module module, GradientCheckUtil.TensorFunction fn, Tensor... inputs) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result;
        module.setRequiresGrad(false);
        try {
            result = GradientCheckUtil.gradCheck(fn, Arrays.asList(inputs), EPSILON, TOLERANCE);
        } finally {
            module.setRequiresGrad(true);
        }
        assertTrue(name + " " + what + " input gradients: " + result, result.passed());
        assertEquals(name + " " + what + " input check leaves the graph as it was", before, graph.size());
    }

    /** The gradients of every parameter of {@code module} for {@code loss}. */
    private void checkParameters(String what, Module module, Supplier<Tensor> loss) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        GradientCheckUtil.GradientCheckResult result = GradientCheckUtil.gradCheckParameters(loss, module,
            module.parameters(), EPSILON, TOLERANCE, MAX_ELEMENTS);
        assertTrue(name + " " + what + " parameter gradients: " + result, result.passed());
        assertEquals(name + " " + what + " parameter check leaves the graph as it was", before, graph.size());
    }

    /** {@code sum(hidden * hiddenProbe) + sum(cell * cellProbe)} of one LSTM step; the states are closed. */
    private static Tensor lstmCellLoss(LSTMCell cell, Tensor x, Tensor h, Tensor c, Tensor hiddenProbe,
                                       Tensor cellProbe) {
        LSTMCell.LSTMCellOutput out = cell.forwardWithCell(x, h, c);
        Tensor cellState = out.cell();
        try (Tensor hiddenTerm = probeSum(out.hidden(), hiddenProbe);
             Tensor cellTerm = probeSum(cellState, cellProbe)) {
            return TensorOps.add(hiddenTerm, cellTerm);
        } finally {
            cellState.close();
        }
    }

    /** {@code sum(output * probe)} as a new scalar; {@code output} is closed. */
    static Tensor probeSum(Tensor output, Tensor probe) {
        try (Tensor out = output;
             Tensor weighted = TensorOps.mul(out, probe)) {
            return TensorOps.sum(weighted);
        }
    }

    // ------------------------------------------------------------------ setup

    /** Initializes a module on the backend from the seed, gives its biases seeded values and puts it in eval mode. */
    private void prepare(Module module, Initializer init) {
        init.initialize(backend, new Random(SEED));
        for (Map.Entry<String, Parameter> e : module.namedParameters().entrySet()) {
            if (e.getKey().endsWith("bias")) {
                Parameter p = e.getValue();
                store(p, TrainParity.uniform(SEED + e.getKey().hashCode(), (int) p.numElements(), -0.5f, 0.5f));
            }
        }
        module.eval();
    }

    /** Overwrites a parameter's values; an assignment, so it runs under {@code noGrad} (in-place guard). */
    private static void store(Parameter p, float[] values) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            GpuOptimizerOps.storeFloats(p.tensor(), values);
        }
    }

    /** A new tensor of {@code dims} uniform in [-1, 1] from {@code seed}, closed after the test. */
    private Tensor tensor(long seed, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        try (Tensor flat = Tensor.fromArray(backend, TrainParity.uniform(seed, n, -1f, 1f))) {
            return keep(flat.reshape(dims));
        }
    }

    private <T extends AutoCloseable> T keep(T resource) {
        owned.add(resource);
        return resource;
    }

    /** Initializes a freshly built module on a backend. */
    private interface Initializer {
        void initialize(ComputeBackend backend, Random random);
    }
}
