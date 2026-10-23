package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
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
import java.util.Random;

import static org.junit.Assert.*;

public class CheckpointFunctionTest {

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
    public void testCheckpointFunctionExists() {
        assertNotNull(CheckpointFunction.class);
    }

    @Test
    public void testCheckpointBasic() {
        assumeGpuAvailable();
        
        Tensor input = Tensor.rand(backend, Shape.of(4, 4)).withRequiresGrad(true);
        
        Tensor output = CheckpointFunction.checkpoint(
                t -> TensorOps.mul(t, t),
                input
        );
        
        assertNotNull("Output should not be null", output);
        assertEquals("Output shape should match input", input.shape(), output.shape());
        
        input.close();
        output.close();
    }

    @Test
    public void testCheckpointPreservesForwardResult() {
        assumeGpuAvailable();
        
        Tensor input = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f, 4.0f})
                .reshape(Shape.of(2, 2))
                .withRequiresGrad(true);
        
        float[] inputData = input.toFloatArray();
        
        Tensor checkpointOutput = CheckpointFunction.checkpoint(
                t -> TensorOps.add(t, t),
                input
        );
        
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            Tensor directOutput = TensorOps.add(input, input);
            
            float[] checkpointData = checkpointOutput.toFloatArray();
            float[] directData = directOutput.toFloatArray();
            
            assertArrayEquals("Checkpoint should produce same result as direct computation",
                    directData, checkpointData, 1e-5f);
            
            directOutput.close();
        }
        
        input.close();
        checkpointOutput.close();
    }

    @Test
    public void testCheckpointSequential() {
        assumeGpuAvailable();
        
        Tensor input = Tensor.rand(backend, Shape.of(4, 4)).withRequiresGrad(true);
        
        List<CheckpointFunction.CheckpointFn> functions = Arrays.asList(
                t -> TensorOps.add(t, Tensor.ones(backend, t.shape())),
                t -> TensorOps.mul(t, Tensor.fromArray(backend, new float[]{2.0f}).reshape(Shape.scalar()))
        );
        
        Tensor output = CheckpointFunction.checkpointSequential(functions, input);
        
        assertNotNull("Sequential output should not be null", output);
        
        input.close();
        output.close();
    }

    @Test
    public void testEstimateMemorySavings() {
        int numLayers = 12;
        long activationSizeBytes = 1024 * 1024; // 1 MB
        int checkpointEvery = 3;
        
        CheckpointFunction.MemoryStats stats = CheckpointFunction.estimateMemorySavings(
                numLayers, activationSizeBytes, checkpointEvery);
        
        assertNotNull("Memory stats should not be null", stats);
        assertTrue("Saved activations should be positive", stats.savedActivationsBytes() > 0);
        assertTrue("Peak memory should be positive", stats.peakMemoryBytes() > 0);
        assertTrue("Number of checkpoints should be positive", stats.numCheckpoints() > 0);
        
        System.out.println("Memory stats: " + stats);
    }

    @Test
    public void testCheckpointSegments() {
        assumeGpuAvailable();
        
        Tensor input = Tensor.rand(backend, Shape.of(4, 4)).withRequiresGrad(true);
        
        List<CheckpointFunction.CheckpointFn> segments = Arrays.asList(
                t -> TensorOps.add(t, Tensor.ones(backend, t.shape())),
                t -> TensorOps.add(t, Tensor.ones(backend, t.shape())),
                t -> TensorOps.add(t, Tensor.ones(backend, t.shape()))
        );
        
        Tensor output = CheckpointFunction.checkpointSegments(segments, 2, input);
        
        assertNotNull("Segments output should not be null", output);
        
        input.close();
        output.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCheckpointNullFunction() {
        assumeGpuAvailable();
        Tensor input = Tensor.zeros(backend, Shape.of(2, 2));
        CheckpointFunction.checkpoint(null, input);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCheckpointNullInput() {
        CheckpointFunction.checkpoint(t -> t, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCheckpointSequentialEmptyList() {
        // Pass null input - validation of empty list happens before input is checked
        CheckpointFunction.checkpointSequential(Arrays.asList(), null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCheckpointSequentialNullList() {
        // Pass null input - validation of null list happens before input is checked
        CheckpointFunction.checkpointSequential(null, null);
    }

    @Test
    public void testWrapCheckpoint() {
        assumeGpuAvailable();
        
        java.util.function.Function<Tensor, Tensor> wrapped = 
                CheckpointFunction.wrapCheckpoint(t -> TensorOps.mul(t, t));
        
        assertNotNull("Wrapped function should not be null", wrapped);
        
        Tensor input = Tensor.rand(backend, Shape.of(2, 2)).withRequiresGrad(true);
        Tensor output = wrapped.apply(input);
        
        assertNotNull("Output from wrapped function should not be null", output);
        
        input.close();
        output.close();
    }

    // ------------------------------------------------------------------ replay and Autograd.grad (host backend)

    @Test
    public void replayableDrawsOutsideACheckpoint() {
        int[] draws = {0};
        assertEquals(Integer.valueOf(7), CheckpointFunction.replayable(() -> {
            draws[0]++;
            return 7;
        }));
        assertEquals(1, draws[0]);
    }

    @Test
    public void recomputeReplaysTheForwardDraws() {
        ComputeBackend host = TrainParity.hostBackend();
        ComputationGraph graph = ComputationGraph.current();
        graph.clear();
        Random rng = new Random(1234);
        int[] draws = {0};
        CheckpointFunction.CheckpointFn segment = t -> {
            float[] mask = CheckpointFunction.replayable(() -> {
                draws[0]++;
                return dropoutMask(rng, t.shape().dim(0));
            });
            try (Tensor m = Tensor.fromArray(t.backend(), mask)) {
                return AutogradRecorderTest.mul(t, m);
            }
        };
        float[] values = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        try (Tensor x = leaf(host, values)) {
            Tensor y = CheckpointFunction.checkpoint(segment, x);
            float[] forward = y.toFloatArray();
            Tensor loss = AutogradRecorderTest.sum(y);
            graph.backward(loss);
            assertEquals("the recompute replays the taped mask instead of drawing", 1, draws[0]);
            float[] gx = x.grad().toFloatArray();
            for (int i = 0; i < values.length; i++) {
                assertEquals("gradient " + i + " is the forward mask", forward[i] / values[i], gx[i], 0f);
            }
            assertEquals(0, graph.size());
            loss.close();
            graph.zeroGrad();
        } finally {
            graph.clear();
            host.close();
        }
    }

    @Test
    public void recomputeDrawingMoreThanTheForwardThrows() {
        ComputeBackend host = TrainParity.hostBackend();
        ComputationGraph graph = ComputationGraph.current();
        graph.clear();
        int[] calls = {0};
        CheckpointFunction.CheckpointFn segment = t -> {
            int count = calls[0]++ == 0 ? 1 : 2;
            float scale = 1f;
            for (int i = 0; i < count; i++) {
                scale *= CheckpointFunction.replayable(() -> 2f);
            }
            try (Tensor s = Tensor.fromArray(t.backend(), filled(t.shape().dim(0), scale))) {
                return AutogradRecorderTest.mul(t, s);
            }
        };
        try (Tensor x = leaf(host, new float[]{1f, 2f})) {
            Tensor y = CheckpointFunction.checkpoint(segment, x);
            Tensor loss = AutogradRecorderTest.sum(y);
            try {
                graph.backward(loss);
                fail("a recompute that draws more than its forward pass is not deterministic");
            } catch (RuntimeException expected) {
                assertTrue(causedBy(expected, IllegalStateException.class));
            }
            assertNull(x.grad());
            graph.release(loss);
            loss.close();
            assertEquals(0, graph.size());
        } finally {
            graph.clear();
            host.close();
        }
    }

    @Test
    public void gradThroughACheckpointThrows() {
        ComputeBackend host = TrainParity.hostBackend();
        ComputationGraph graph = ComputationGraph.current();
        graph.clear();
        try (Tensor x = leaf(host, new float[]{1f, 2f, 3f})) {
            Tensor y = CheckpointFunction.checkpoint(t -> AutogradRecorderTest.mul(t, t), x);
            Tensor loss = AutogradRecorderTest.sum(y);
            int recorded = graph.size();
            try {
                Autograd.grad(loss, Collections.singletonList(x), null, true, false);
                fail("Autograd.grad must refuse a checkpointed segment");
            } catch (UnsupportedOperationException expected) {
                assertEquals(recorded, graph.size());
            }
            try {
                graph.backward(loss, null, true, true);
                fail("createGraph must refuse a checkpointed segment");
            } catch (UnsupportedOperationException expected) {
                assertNull(x.grad());
            }
            graph.release(loss);
            loss.close();
            assertEquals(0, graph.size());
        } finally {
            graph.clear();
            host.close();
        }
    }

    private static Tensor leaf(ComputeBackend backend, float[] values) {
        try (Tensor t = Tensor.fromArray(backend, values)) {
            return t.withRequiresGrad(true);
        }
    }

    /** Inverted-dropout mask with keep probability 0.5: each entry 0 or 2. */
    private static float[] dropoutMask(Random rng, int n) {
        float[] mask = new float[n];
        for (int i = 0; i < n; i++) {
            mask[i] = rng.nextFloat() < 0.5f ? 0f : 2f;
        }
        return mask;
    }

    private static float[] filled(int n, float value) {
        float[] out = new float[n];
        Arrays.fill(out, value);
        return out;
    }

    private static boolean causedBy(Throwable t, Class<? extends Throwable> type) {
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (type.isInstance(c)) {
                return true;
            }
        }
        return false;
    }
}
