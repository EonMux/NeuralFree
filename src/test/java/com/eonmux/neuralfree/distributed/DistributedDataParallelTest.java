package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorValues;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class DistributedDataParallelTest {

    @Test
    public void testConstructionWithValidParams() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend);
        
        assertEquals(0, ddp.rank());
        assertEquals(2, ddp.worldSize());
        assertTrue(ddp.isDistributed());
        assertNotNull(ddp.module());
        assertNotNull(ddp.backend());
        
        ddp.close();
        mockBackend.close();
    }

    @Test
    public void testConstructionWithBucketing() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend, true, 10);
        
        assertNotNull(ddp);
        
        ddp.close();
        mockBackend.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullModuleThrows() {
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        new DistributedDataParallel(null, mockBackend);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullBackendThrows() {
        Module mockModule = new MockModule();
        new DistributedDataParallel(mockModule, null);
    }

    @Test
    public void testTrainEvalModes() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend);
        
        ddp.train();
        ddp.eval();
        
        ddp.close();
        mockBackend.close();
    }

    @Test
    public void testZeroGrad() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend);
        
        ddp.zeroGrad();
        
        ddp.close();
        mockBackend.close();
    }

    @Test
    public void testStateDict() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend);
        
        Map<String, Object> stateDict = ddp.stateDict();
        assertNotNull(stateDict);
        
        ddp.close();
        mockBackend.close();
    }

    @Test
    public void testOperationsAfterCloseThrow() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend);
        
        ddp.close();
        
        try {
            ddp.rank();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
        
        try {
            ddp.module();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
    }

    @Test
    public void testSingleProcessNotDistributed() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 1);
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend);
        
        assertFalse(ddp.isDistributed());
        
        ddp.close();
        mockBackend.close();
    }

    @Test
    public void testCloseIsIdempotent() throws Exception {
        Module mockModule = new MockModule();
        DistributedBackend mockBackend = new MockDistributedBackend(0, 2);
        DistributedDataParallel ddp = new DistributedDataParallel(mockModule, mockBackend);
        
        ddp.close();
        ddp.close();
        ddp.close();
        
        mockBackend.close();
    }

    /**
     * backward(loss) backpropagates the replica's loss, then all-reduces each parameter gradient once, with grad mode
     * off during the all-reduce and restored afterwards; the replica's graph is released.
     */
    @Test
    public void testBackwardBackpropagatesThenReducesWithoutRecording() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        LinearReplica replica = new LinearReplica(host);
        MockDistributedBackend backend = new MockDistributedBackend(0, 2);
        DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, false, 25);
        int graphBefore = ComputationGraph.current().size();
        try (Tensor x = Tensor.fromArray(host, new float[][]{{1f, 2f}, {3f, -1f}})) {
            Tensor out = ddp.forward(x);
            assertTrue("forward records in grad mode", out.requiresGrad());
            Tensor loss = TensorOps.sum(out);
            ddp.backward(loss);
            loss.close();
            out.close();
            assertArrayEquals("d(sum x w)/dw = column sums of x", new float[]{4f, 1f},
                replica.weight.grad().toFloatArray(), 0.0f);
            assertEquals("one all-reduce per bucket", Collections.singletonList(Boolean.FALSE),
                backend.reduceGradModes());
            assertTrue("grad mode restored", ComputationGraph.isGradEnabled());
            assertEquals("graph released", graphBefore, ComputationGraph.current().size());
        } finally {
            ddp.close();
            backend.close();
            replica.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /** step(optimizer) all-reduces the gradients and steps the optimizer, both with grad mode off. */
    @Test
    public void testStepReducesAndStepsWithoutRecording() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        LinearReplica replica = new LinearReplica(host);
        MockDistributedBackend backend = new MockDistributedBackend(0, 2);
        DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
        RecordingOptimizer optimizer = new RecordingOptimizer(replica);
        try (Tensor x = Tensor.fromArray(host, new float[][]{{1f, 2f}})) {
            Tensor out = ddp.forward(x);
            Tensor loss = TensorOps.sum(out);
            loss.backward();
            loss.close();
            out.close();
            ddp.step(optimizer);
            assertEquals(Collections.singletonList(Boolean.FALSE), backend.reduceGradModes());
            assertEquals(Collections.singletonList(Boolean.FALSE), optimizer.stepGradModes);
            assertTrue("grad mode restored", ComputationGraph.isGradEnabled());
        } finally {
            ddp.close();
            backend.close();
            replica.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNonPositiveBucketSizeThrows() {
        new DistributedDataParallel(new MockModule(), new MockDistributedBackend(0, 2), true, 0);
    }

    /**
     * A trainable parameter without a local gradient contributes zeros and a zero usage flag to its bucket's single
     * all-reduce (bucket order [b, a], then the flags), and takes the mean gradient when the other rank produced one.
     */
    @Test
    public void testParameterWithoutLocalGradientJoinsTheBucketReduction() {
        ComputeBackend host = TrainParity.hostBackend();
        TwoParamReplica replica = new TwoParamReplica(host, 1f);
        PeerRankBackend backend = new PeerRankBackend(0, new float[]{4f, 6f, 3f, 4f, 1f, 1f}, 0f);
        try {
            DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                replica.a.setGrad(Tensor.fromArray(host, new float[]{1f, 2f}));
            }
            ddp.synchronizeGradients();
            assertEquals("one all-reduce for the bucket", 1, backend.reduced.size());
            assertArrayEquals("[b grad (zeros), a grad, b flag, a flag]", new float[]{0f, 0f, 1f, 2f, 0f, 1f},
                backend.reduced.get(0), 0f);
            assertArrayEquals("mean of a", new float[]{2f, 3f}, replica.a.grad().toFloatArray(), 0f);
            assertArrayEquals("mean of b, produced by the other rank only", new float[]{2f, 3f},
                replica.b.grad().toFloatArray(), 0f);
            ddp.close();
        } finally {
            replica.close();
            TrainGpu.release(host);
        }
    }

    /** A parameter no rank produced a gradient for stays without one, as with PyTorch's find_unused_parameters. */
    @Test
    public void testParameterUnusedOnEveryRankKeepsNoGradient() {
        ComputeBackend host = TrainParity.hostBackend();
        TwoParamReplica replica = new TwoParamReplica(host, 1f);
        PeerRankBackend backend = new PeerRankBackend(0, new float[]{0f, 0f, 3f, 4f, 0f, 1f}, 0f);
        try {
            DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                replica.a.setGrad(Tensor.fromArray(host, new float[]{1f, 2f}));
            }
            ddp.synchronizeGradients();
            assertArrayEquals(new float[]{2f, 3f}, replica.a.grad().toFloatArray(), 0f);
            assertNull("unused on every rank", replica.b.grad());
            ddp.close();
        } finally {
            replica.close();
            TrainGpu.release(host);
        }
    }

    /** Construction overwrites a non-zero rank's parameters with rank 0's through one broadcast. */
    @Test
    public void testConstructionTakesRankZeroParameters() {
        ComputeBackend host = TrainParity.hostBackend();
        TwoParamReplica replica = new TwoParamReplica(host, 1f);
        PeerRankBackend backend = new PeerRankBackend(1, new float[0], 5f);
        try {
            DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
            assertEquals("one broadcast of [a, b]", Collections.singletonList(4), backend.broadcastSizes);
            assertArrayEquals(new float[]{5f, 5f}, replica.a.tensor().toFloatArray(), 0f);
            assertArrayEquals(new float[]{5f, 5f}, replica.b.tensor().toFloatArray(), 0f);
            assertTrue("still trainable leaves", replica.a.tensor().requiresGrad());
            ddp.close();
        } finally {
            replica.close();
            TrainGpu.release(host);
        }
    }

    /**
     * In training mode each forward first takes rank 0's persistent buffers (one broadcast of the persistent ones
     * only); in evaluation mode it does not.
     */
    @Test
    public void testForwardBroadcastsPersistentBuffersInTrainingMode() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        BufferedReplica replica = new BufferedReplica(host);
        PeerRankBackend backend = new PeerRankBackend(1, new float[0], 9f);
        try (Tensor x = Tensor.fromArray(host, new float[]{1f})) {
            DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
            assertEquals("construction: weight and persistent buffer", Collections.singletonList(3),
                backend.broadcastSizes);
            assertArrayEquals(new float[]{9f, 9f}, replica.running().toFloatArray(), 0f);
            assertArrayEquals("non-persistent buffer untouched", new float[]{-1f, -1f, -1f},
                replica.cache().toFloatArray(), 0f);

            replica.setRunning(new float[]{1f, 1f});
            ddp.train();
            ddp.forward(x).close();
            assertEquals(Arrays.asList(3, 2), backend.broadcastSizes);
            assertArrayEquals(new float[]{9f, 9f}, replica.running().toFloatArray(), 0f);

            replica.setRunning(new float[]{1f, 1f});
            ddp.eval();
            ddp.forward(x).close();
            assertEquals("no broadcast in evaluation mode", Arrays.asList(3, 2), backend.broadcastSizes);
            assertArrayEquals(new float[]{1f, 1f}, replica.running().toFloatArray(), 0f);
            ddp.close();
        } finally {
            replica.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * Two ranks over {@link SharedMemoryBackend}: the replicas start from different weights and backpropagate into
     * different parameters (rank 0 into {@code a}, rank 1 into {@code b}). After construction both hold rank 0's
     * weights, and after the reduction both hold {@code a.grad = [2, 4] / 2} and {@code b.grad = [6, 8] / 2}.
     */
    @Test
    public void testRanksWithDifferentUsedParametersReduceConsistently() throws Exception {
        String prefix = "nf_ddp_unused_" + System.nanoTime();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            List<Future<float[][]>> ranks = new ArrayList<>();
            for (int r = 0; r < 2; r++) {
                int rank = r;
                ranks.add(pool.submit(() -> runReplica(rank, prefix)));
            }
            for (Future<float[][]> rank : ranks) {
                float[][] result;
                try {
                    result = rank.get(120, TimeUnit.SECONDS);
                } catch (ExecutionException e) {
                    throw new AssertionError(e.getCause());
                }
                assertArrayEquals("rank 0's initial weights", new float[]{1f, 1f}, result[0], 0f);
                assertArrayEquals("a.grad", new float[]{1f, 2f}, result[1], 0f);
                assertArrayEquals("b.grad", new float[]{3f, 4f}, result[2], 0f);
            }
        } finally {
            pool.shutdownNow();
        }
    }

    /** One rank of {@link #testRanksWithDifferentUsedParametersReduceConsistently}: [a after construction, a.grad, b.grad]. */
    private static float[][] runReplica(int rank, String prefix) {
        ComputeBackend host = TrainParity.hostBackend();
        TwoParamReplica replica = new TwoParamReplica(host, rank + 1f);
        try (SharedMemoryBackend backend = new SharedMemoryBackend(rank, 2, prefix)) {
            DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
            float[] initial = replica.a.tensor().toFloatArray();
            Parameter used = rank == 0 ? replica.a : replica.b;
            try (Tensor x = Tensor.fromArray(host, rank == 0 ? new float[]{2f, 4f} : new float[]{6f, 8f})) {
                Tensor product = TensorOps.mul(used.tensor(), x);
                Tensor loss = TensorOps.sum(product);
                loss.backward();
                loss.close();
                product.close();
            }
            ddp.synchronizeGradients();
            float[][] result = {initial, replica.a.grad().toFloatArray(), replica.b.grad().toFloatArray()};
            ddp.close();
            return result;
        } finally {
            replica.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * Gradients of a model converted to a half-precision dtype are exchanged as FLOAT32 and come back in that dtype:
     * {@code a}'s own gradient and {@code b}'s gradient, produced by the other rank only, take the mean gradient.
     */
    @Test
    public void testHalfPrecisionGradientsAreReducedInTheirDType() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            ComputeBackend host = TrainParity.hostBackend();
            TwoParamReplica replica = new TwoParamReplica(host, 1f);
            replica.to(dtype);
            PeerRankBackend backend = new PeerRankBackend(0, new float[]{4f, 6f, 3f, 4f, 1f, 1f}, 0f);
            try {
                DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                     Tensor floats = Tensor.fromArray(host, new float[]{1f, 2f})) {
                    replica.a.setGrad(floats.to(dtype));
                }
                ddp.synchronizeGradients();
                assertArrayEquals(dtype + ": [b grad (zeros), a grad, b flag, a flag]",
                    new float[]{0f, 0f, 1f, 2f, 0f, 1f}, backend.reduced.get(0), 0f);
                assertEquals(dtype, replica.a.grad().dtype());
                assertArrayEquals(dtype + ": mean of a", new float[]{2f, 3f},
                    TensorValues.toFloats(replica.a.grad()), 0f);
                assertEquals(dtype, replica.b.grad().dtype());
                assertArrayEquals(dtype + ": mean of b", new float[]{2f, 3f},
                    TensorValues.toFloats(replica.b.grad()), 0f);
                ddp.close();
            } finally {
                replica.close();
                TrainGpu.release(host);
            }
        }
    }

    /**
     * Persistent buffers of any dtype (an INT32 counter and a FLOAT16 running value) take rank 0's values at
     * construction and on every training-mode forward, and keep their dtypes.
     */
    @Test
    public void testForwardBroadcastsBuffersOfEveryDType() {
        ComputationGraph.current().clear();
        ComputeBackend host = TrainParity.hostBackend();
        MixedBufferReplica replica = new MixedBufferReplica(host);
        PeerRankBackend backend = new PeerRankBackend(1, new float[0], 9f);
        try (Tensor x = Tensor.fromArray(host, new float[]{1f})) {
            DistributedDataParallel ddp = new DistributedDataParallel(replica, backend, true, 25);
            assertEquals("construction: weight, steps and running", Collections.singletonList(5),
                backend.broadcastSizes);
            replica.assertBuffers(9f);

            ddp.train();
            ddp.forward(x).close();
            assertEquals(Arrays.asList(5, 4), backend.broadcastSizes);
            replica.assertBuffers(9f);
            ddp.close();
        } finally {
            replica.close();
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }
    }

    /**
     * A parameter {@code w} = [0] and the persistent buffers {@code steps} (INT32 [0, 0]) and {@code running}
     * (FLOAT16 [0, 0]); the forward pass returns {@code w}'s value as a new tensor.
     */
    static final class MixedBufferReplica extends Module {
        final Parameter weight;

        MixedBufferReplica(ComputeBackend host) {
            weight = registerParameter("w", new Parameter(Tensor.fromArray(host, new float[]{0f}), "w"));
            registerBuffer("steps", Tensor.fromArray(host, new int[]{0, 0}));
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor floats = Tensor.fromArray(host, new float[]{0f, 0f})) {
                registerBuffer("running", floats.to(DType.FLOAT16));
            }
        }

        /** Both buffers hold {@code value} in their own dtypes. */
        void assertBuffers(float value) {
            assertEquals(DType.INT32, buffer("steps").dtype());
            assertArrayEquals(new int[]{(int) value, (int) value}, buffer("steps").toIntArray());
            assertEquals(DType.FLOAT16, buffer("running").dtype());
            assertArrayEquals(new float[]{value, value}, TensorValues.toFloats(buffer("running")), 0f);
        }

        @Override
        public Tensor forward(Tensor input) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                return Tensor.fromArray(input.backend(), weight.tensor().toFloatArray());
            }
        }
    }

    /** Two vector parameters {@code a} and {@code b} of two elements, both filled with {@code init}. */
    static final class TwoParamReplica extends Module {
        final Parameter a;
        final Parameter b;

        TwoParamReplica(ComputeBackend host, float init) {
            a = registerParameter("a", new Parameter(Tensor.fromArray(host, new float[]{init, init}), "a"));
            b = registerParameter("b", new Parameter(Tensor.fromArray(host, new float[]{init, init}), "b"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    /**
     * A parameter {@code w} = [0], a persistent buffer {@code running} = [0, 0] and a non-persistent buffer
     * {@code cache} = [-1, -1, -1]; the forward pass returns {@code w}'s value as a new tensor.
     */
    static final class BufferedReplica extends Module {
        final Parameter weight;

        BufferedReplica(ComputeBackend host) {
            weight = registerParameter("w", new Parameter(Tensor.fromArray(host, new float[]{0f}), "w"));
            registerBuffer("running", Tensor.fromArray(host, new float[]{0f, 0f}));
            registerBuffer("cache", Tensor.fromArray(host, new float[]{-1f, -1f, -1f}), false);
        }

        Tensor running() {
            return buffer("running");
        }

        Tensor cache() {
            return buffer("cache");
        }

        void setRunning(float[] values) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                CollectiveTensors.writeBack(running(), values);
            }
        }

        @Override
        public Tensor forward(Tensor input) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                return Tensor.fromArray(input.backend(), weight.tensor().toFloatArray());
            }
        }
    }

    /**
     * One rank of two whose peer is simulated: an all-reduce (MEAN) averages the caller's values with
     * {@code peerValues} and records the caller's values; a broadcast from another rank fills the tensor with
     * {@code broadcastFill} and records its element count.
     */
    static final class PeerRankBackend implements DistributedBackend {
        private final int rank;
        private final float[] peerValues;
        private final float broadcastFill;
        final List<float[]> reduced = new ArrayList<>();
        final List<Integer> broadcastSizes = new ArrayList<>();

        PeerRankBackend(int rank, float[] peerValues, float broadcastFill) {
            this.rank = rank;
            this.peerValues = peerValues.clone();
            this.broadcastFill = broadcastFill;
        }

        @Override
        public int rank() {
            return rank;
        }

        @Override
        public int worldSize() {
            return 2;
        }

        @Override
        public void allReduce(Tensor tensor, ReduceOp op) {
            assertEquals(ReduceOp.MEAN, op);
            float[] mine = tensor.toFloatArray();
            reduced.add(mine.clone());
            assertEquals("peer values for every element", peerValues.length, mine.length);
            float[] mean = new float[mine.length];
            for (int i = 0; i < mine.length; i++) {
                mean[i] = (mine[i] + peerValues[i]) / 2f;
            }
            CollectiveTensors.writeBack(tensor, mean);
        }

        @Override
        public void broadcast(Tensor tensor, int srcRank) {
            broadcastSizes.add(Math.toIntExact(tensor.numElements()));
            if (srcRank != rank) {
                float[] fill = new float[Math.toIntExact(tensor.numElements())];
                Arrays.fill(fill, broadcastFill);
                CollectiveTensors.writeBack(tensor, fill);
            }
        }

        @Override
        public void allGather(List<Tensor> output, Tensor input) {
            throw new UnsupportedOperationException("not used by DistributedDataParallel");
        }

        @Override
        public void barrier() {
        }

        @Override
        public boolean isInitialized() {
            return true;
        }

        @Override
        public void close() {
        }
    }

    /** {@code x w} with one parameter {@code w} of shape [2, 1], initialised to {@code [1, -1]}. */
    static final class LinearReplica extends Module {
        final Parameter weight;

        LinearReplica(ComputeBackend b) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, new float[][]{{1f}, {-1f}}),
                "weight"));
        }

        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    /** Records the grad mode of every step. */
    static final class RecordingOptimizer extends Optimizer {
        final List<Boolean> stepGradModes = new ArrayList<>();

        RecordingOptimizer(Module model) {
            super(model.parameters(), new HashMap<>());
        }

        @Override
        public void step() {
            stepGradModes.add(ComputationGraph.isGradEnabled());
        }
    }

    static class MockModule extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    static class MockDistributedBackend implements DistributedBackend {
        private final int rank;
        private final int worldSize;
        private final List<Boolean> reduceGradModes = new ArrayList<>();
        private boolean closed = false;

        MockDistributedBackend(int rank, int worldSize) {
            this.rank = rank;
            this.worldSize = worldSize;
        }

        @Override
        public int rank() { 
            if (closed) throw new IllegalStateException("Closed");
            return rank; 
        }

        @Override
        public int worldSize() { 
            if (closed) throw new IllegalStateException("Closed");
            return worldSize; 
        }

        /** The grad mode of every all-reduce call, in order. */
        List<Boolean> reduceGradModes() {
            return new ArrayList<>(reduceGradModes);
        }

        @Override
        public void allReduce(Tensor tensor, ReduceOp op) {
            if (closed) throw new IllegalStateException("Closed");
            reduceGradModes.add(ComputationGraph.isGradEnabled());
        }

        @Override
        public void broadcast(Tensor tensor, int srcRank) {
            if (closed) throw new IllegalStateException("Closed");
        }

        @Override
        public void allGather(List<Tensor> output, Tensor input) {
            if (closed) throw new IllegalStateException("Closed");
        }

        @Override
        public void barrier() {
            if (closed) throw new IllegalStateException("Closed");
        }

        @Override
        public boolean isInitialized() { return !closed; }

        @Override
        public void close() { closed = true; }
    }
}
