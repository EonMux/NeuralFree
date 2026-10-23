package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.nn.Parameter;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class FederatedTrainerTest {

    @Test
    public void testConstructionWithValidParams() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        
        FederatedTrainer trainer = new FederatedTrainer(
            mockModel, mockOptimizer, 5, 10, 3, 0.01f
        );
        
        assertEquals(5, trainer.numClients());
        assertEquals(10, trainer.numRounds());
        assertEquals(FederatedTrainer.FederatedStrategy.FED_AVG, trainer.strategy());
        assertNotNull(trainer.globalModel());
        
        trainer.close();
    }

    @Test
    public void testConstructionWithFedProx() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        
        FederatedTrainer trainer = new FederatedTrainer(
            mockModel, mockOptimizer, 3, 5, 2, 0.01f, 
            FederatedTrainer.FederatedStrategy.FED_PROX, 0.1f
        );
        
        assertEquals(FederatedTrainer.FederatedStrategy.FED_PROX, trainer.strategy());
        
        trainer.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullModelThrows() {
        Optimizer mockOptimizer = new MockOptimizer(new MockModule());
        new FederatedTrainer(null, mockOptimizer, 5, 10, 3, 0.01f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullOptimizerThrows() {
        Module mockModel = new MockModule();
        new FederatedTrainer(mockModel, null, 5, 10, 3, 0.01f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidNumClientsZeroThrows() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        new FederatedTrainer(mockModel, mockOptimizer, 0, 10, 3, 0.01f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidNumClientsNegativeThrows() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        new FederatedTrainer(mockModel, mockOptimizer, -1, 10, 3, 0.01f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidNumRoundsThrows() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        new FederatedTrainer(mockModel, mockOptimizer, 5, 0, 3, 0.01f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidLocalEpochsThrows() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        new FederatedTrainer(mockModel, mockOptimizer, 5, 10, 0, 0.01f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidLearningRateZeroThrows() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        new FederatedTrainer(mockModel, mockOptimizer, 5, 10, 3, 0.0f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidLearningRateNegativeThrows() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        new FederatedTrainer(mockModel, mockOptimizer, 5, 10, 3, -0.01f);
    }

    @Test
    public void testOperationsAfterCloseThrow() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        FederatedTrainer trainer = new FederatedTrainer(mockModel, mockOptimizer, 5, 10, 3, 0.01f);
        
        trainer.close();
        
        try {
            trainer.numClients();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
        
        try {
            trainer.globalModel();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
    }

    @Test
    public void testCloseIsIdempotent() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        FederatedTrainer trainer = new FederatedTrainer(mockModel, mockOptimizer, 5, 10, 3, 0.01f);
        
        trainer.close();
        trainer.close();
        trainer.close();
    }

    @Test
    public void testFederatedStrategies() {
        assertEquals(2, FederatedTrainer.FederatedStrategy.values().length);
        assertNotNull(FederatedTrainer.FederatedStrategy.FED_AVG);
        assertNotNull(FederatedTrainer.FederatedStrategy.FED_PROX);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTrainWithMismatchedDataSizeThrows() {
        Module mockModel = new MockModule();
        Optimizer mockOptimizer = new MockOptimizer(mockModel);
        FederatedTrainer trainer = new FederatedTrainer(mockModel, mockOptimizer, 3, 1, 1, 0.01f);
        
        List<Tensor> clientData = new ArrayList<>();
        List<Tensor> clientLabels = new ArrayList<>();
        clientData.add(null);
        
        trainer.train(clientData, clientLabels);
        
        trainer.close();
    }

    /**
     * One FedAvg round with SGD (learning rate 0.5) over two clients of one sample each, loss {@code (x w - y)^2}:
     * client 0 ({@code x = [1, 1], y = 3}) moves {@code w = [1, 0]} to {@code [3, 2]}, client 1 ({@code x = [2, 0],
     * y = 0}) to {@code [-3, 0]}, so the average is {@code [0, 1]} exactly. Every local step leaves the graph at its
     * size before training and no gradient is left behind.
     */
    @Test
    public void testFedAvgRoundRecordsLocallyAndLeavesNoNodes() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        LinearModel model = new LinearModel(host);
        List<Tensor> data = new ArrayList<>();
        List<Tensor> labels = new ArrayList<>();
        data.add(Tensor.fromArray(host, new float[][]{{1f, 1f}}));
        data.add(Tensor.fromArray(host, new float[][]{{2f, 0f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{3f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{0f}}));
        FederatedTrainer trainer = new FederatedTrainer(model, new SGD(model.parameters(), 0.5f), 2, 1, 1, 0.5f);
        int graphBefore = ComputationGraph.current().size();
        try {
            FederatedTrainer.FederatedResult result = trainer.train(data, labels);
            assertArrayEquals("averaged weights", new float[]{0f, 1f}, model.weight.tensor().toFloatArray(), 0.0f);
            assertEquals(1, result.roundLosses.size());
            assertEquals("graph size", graphBefore, ComputationGraph.current().size());
            assertNull("gradients released", model.weight.grad());
        } finally {
            trainer.close();
            for (Tensor t : data) {
                t.close();
            }
            for (Tensor t : labels) {
                t.close();
            }
            model.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * The local loss includes the auxiliary term the forward pass leaves pending: one client ({@code x = [1, 1],
     * y = 3}) with the term {@code sum(w)}, so the gradient at {@code w = [1, 0]} is {@code [-4, -4] + [1, 1]} and SGD
     * with learning rate 0.5 gives {@code [2.5, 1.5]} exactly. The evaluation pass discards its constant term, and no
     * node or gradient is left behind.
     */
    @Test
    public void testLocalLossIncludesTheAuxiliaryTerm() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        AuxiliaryLinearModel model = new AuxiliaryLinearModel(host);
        List<Tensor> data = new ArrayList<>();
        List<Tensor> labels = new ArrayList<>();
        data.add(Tensor.fromArray(host, new float[][]{{1f, 1f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{3f}}));
        FederatedTrainer trainer = new FederatedTrainer(model, new SGD(model.parameters(), 0.5f), 1, 1, 1, 0.5f);
        int graphBefore = ComputationGraph.current().size();
        try {
            trainer.train(data, labels);
            assertArrayEquals("weights", new float[]{2.5f, 1.5f}, model.weight.tensor().toFloatArray(), 0.0f);
            assertEquals("graph size", graphBefore, ComputationGraph.current().size());
            assertNull("gradients released", model.weight.grad());
            assertNull("no auxiliary term left pending", model.takeAuxiliaryLoss());
        } finally {
            trainer.close();
            for (Tensor t : data) {
                t.close();
            }
            for (Tensor t : labels) {
                t.close();
            }
            model.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * Persistent buffers follow FedAvg like the weights: each client starts from the global buffer (0) and adds its
     * first input value in training mode (client 0: 1, client 1: 2), so the global buffer becomes their mean 1.5;
     * the evaluation passes leave it unchanged. The weights average to {@code [0, 1]} as in
     * {@link #testFedAvgRoundRecordsLocallyAndLeavesNoNodes}.
     */
    @Test
    public void testFedAvgAveragesPersistentBuffersFromTheGlobalState() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        CountingLinearModel model = new CountingLinearModel(host);
        List<Tensor> data = new ArrayList<>();
        List<Tensor> labels = new ArrayList<>();
        data.add(Tensor.fromArray(host, new float[][]{{1f, 1f}}));
        data.add(Tensor.fromArray(host, new float[][]{{2f, 0f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{3f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{0f}}));
        FederatedTrainer trainer = new FederatedTrainer(model, new SGD(model.parameters(), 0.5f), 2, 1, 1, 0.5f);
        try {
            trainer.train(data, labels);
            assertArrayEquals("mean of the clients' buffers", new float[]{1.5f}, model.count().toFloatArray(), 0.0f);
            assertArrayEquals("averaged weights", new float[]{0f, 1f}, model.weight.tensor().toFloatArray(), 0.0f);
        } finally {
            trainer.close();
            for (Tensor t : data) {
                t.close();
            }
            for (Tensor t : labels) {
                t.close();
            }
            model.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * FedProx with {@code mu = 1} over {@code (w + d) x}, whose backward pass hands one gradient buffer to both
     * parameters: one client ({@code x = 1, y = 0}) from {@code w = 1, d = 0}, and an optimizer that records the
     * gradients and then moves {@code w} by 1 and {@code d} by 2. The first step has no drift (gradients 2 and 2);
     * the second, at {@code w = d = 2}, has the data gradient 8 for both, plus each parameter's own term only:
     * {@code 8 + (2 - 1) = 9} for {@code w} and {@code 8 + (2 - 0) = 10} for {@code d}.
     */
    @Test
    public void testFedProxGivesParametersSharingAGradientOnlyTheirOwnTerm() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        SumModel model = new SumModel(host);
        List<Tensor> data = new ArrayList<>();
        List<Tensor> labels = new ArrayList<>();
        data.add(Tensor.fromArray(host, new float[][]{{1f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{0f}}));
        RecordingStepOptimizer optimizer = new RecordingStepOptimizer(model, new float[]{1f, 2f});
        FederatedTrainer trainer = new FederatedTrainer(model, optimizer, 1, 1, 2, 0.5f,
            FederatedTrainer.FederatedStrategy.FED_PROX, 1f);
        try {
            trainer.train(data, labels);
            assertEquals("two local steps", 2, optimizer.gradients.size());
            assertArrayEquals("no drift at the first step", new float[]{2f, 2f}, optimizer.gradients.get(0), 0f);
            assertArrayEquals("each parameter's own proximal term", new float[]{9f, 10f},
                optimizer.gradients.get(1), 0f);
            assertNull("gradients released", model.w.grad());
            assertNull("gradients released", model.d.grad());
        } finally {
            trainer.close();
            for (Tensor t : data) {
                t.close();
            }
            for (Tensor t : labels) {
                t.close();
            }
            model.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * A round over a model with an INT32 persistent buffer: the buffer is snapshotted, averaged and restored as state
     * dict values, so it keeps its dtype and value, and the weights average as in
     * {@link #testFedAvgRoundRecordsLocallyAndLeavesNoNodes}.
     */
    @Test
    public void testFedAvgKeepsIntegerPersistentBuffers() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        StepCountedLinearModel model = new StepCountedLinearModel(host);
        List<Tensor> data = new ArrayList<>();
        List<Tensor> labels = new ArrayList<>();
        data.add(Tensor.fromArray(host, new float[][]{{1f, 1f}}));
        data.add(Tensor.fromArray(host, new float[][]{{2f, 0f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{3f}}));
        labels.add(Tensor.fromArray(host, new float[][]{{0f}}));
        FederatedTrainer trainer = new FederatedTrainer(model, new SGD(model.parameters(), 0.5f), 2, 1, 1, 0.5f);
        try {
            trainer.train(data, labels);
            assertEquals(DType.INT32, model.steps().dtype());
            assertArrayEquals(new int[]{7}, model.steps().toIntArray());
            assertArrayEquals("averaged weights", new float[]{0f, 1f}, model.weight.tensor().toFloatArray(), 0.0f);
        } finally {
            trainer.close();
            for (Tensor t : data) {
                t.close();
            }
            for (Tensor t : labels) {
                t.close();
            }
            model.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /** {@code x (w + d)} with the parameters {@code w = [[1]]} and {@code d = [[0]]}. */
    static final class SumModel extends Module {
        final Parameter w;
        final Parameter d;

        SumModel(ComputeBackend b) {
            w = registerParameter("w", new Parameter(Tensor.fromArray(b, new float[][]{{1f}}), "w"));
            d = registerParameter("d", new Parameter(Tensor.fromArray(b, new float[][]{{0f}}), "d"));
        }

        /** While recording, the step scope releases the sum; without recording it is closed here. */
        @Override
        public Tensor forward(Tensor input) {
            Tensor sum = TensorOps.add(w.tensor(), d.tensor());
            if (ComputationGraph.isGradEnabled()) {
                return TensorOps.matmul(input, sum);
            }
            try (Tensor released = sum) {
                return TensorOps.matmul(input, released);
            }
        }
    }

    /**
     * Records the gradients of its parameters (one value each, in parameter order) at every step, then adds
     * {@code moves[i]} to parameter {@code i}.
     */
    static final class RecordingStepOptimizer extends Optimizer {
        final List<float[]> gradients = new ArrayList<>();
        private final float[] moves;

        RecordingStepOptimizer(Module model, float[] moves) {
            super(model.parameters(), new java.util.HashMap<>());
            this.moves = moves.clone();
        }

        @Override
        public void step() {
            float[] step = new float[params.size()];
            for (int i = 0; i < step.length; i++) {
                step[i] = params.get(i).grad().toFloatArray()[0];
            }
            gradients.add(step);
            for (int i = 0; i < step.length; i++) {
                Parameter param = params.get(i);
                float moved = param.tensor().toFloatArray()[0] + moves[i];
                try (Tensor value = Tensor.fromArray(param.backend(), new float[][]{{moved}})) {
                    param.copyFrom(value);
                }
            }
        }
    }

    /** {@code x w} with {@code w = [1, 0]} of shape [2, 1] and an INT32 persistent buffer {@code steps = [7]}. */
    static final class StepCountedLinearModel extends Module {
        final Parameter weight;

        StepCountedLinearModel(ComputeBackend b) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, new float[][]{{1f}, {0f}}),
                "weight"));
            registerBuffer("steps", Tensor.fromArray(b, new int[]{7}));
        }

        Tensor steps() {
            return buffer("steps");
        }

        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    /**
     * {@code x w} with {@code w = [1, 0]} of shape [2, 1] and a persistent buffer {@code count = [0]} to which each
     * forward pass in training mode adds the input's first value.
     */
    static final class CountingLinearModel extends Module {
        final Parameter weight;

        CountingLinearModel(ComputeBackend b) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, new float[][]{{1f}, {0f}}),
                "weight"));
            registerBuffer("count", Tensor.fromArray(b, new float[]{0f}));
        }

        Tensor count() {
            return buffer("count");
        }

        @Override
        public Tensor forward(Tensor input) {
            if (isTraining()) {
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    float[] value = count().toFloatArray();
                    value[0] += input.toFloatArray()[0];
                    GpuOptimizerOps.storeFloats(count(), value);
                }
            }
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    /** {@code x w} with {@code w = [1, 0]} of shape [2, 1]; each forward leaves the auxiliary term {@code sum(w)}. */
    static final class AuxiliaryLinearModel extends Module {
        final Parameter weight;

        AuxiliaryLinearModel(ComputeBackend b) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, new float[][]{{1f}, {0f}}),
                "weight"));
        }

        @Override
        public Tensor forward(Tensor input) {
            setAuxiliaryLoss(TensorOps.sum(weight.tensor()));
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    /** {@code x w} with one parameter {@code w} of shape [2, 1], initialised to {@code [1, 0]}. */
    static final class LinearModel extends Module {
        final Parameter weight;

        LinearModel(ComputeBackend b) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, new float[][]{{1f}, {0f}}),
                "weight"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    static class MockModule extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    static class MockOptimizer extends Optimizer {
        protected MockOptimizer(Module model) {
            super(model.parameters(), new java.util.HashMap<>());
        }

        @Override
        public void step() {
        }
    }
}
