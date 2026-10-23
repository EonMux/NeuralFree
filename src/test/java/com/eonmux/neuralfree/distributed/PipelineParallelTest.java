package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class PipelineParallelTest {

    /** Batch of five rows; every value, product and sum below is a small integer, so all results are exact. */
    private static final float[][] INPUT = {{1f, 2f}, {-1f, 3f}, {2f, 0f}, {0f, -2f}, {3f, 1f}};
    private static final float[][] W0 = {{1f, -1f}, {2f, 1f}};
    private static final float[][] W1 = {{3f}, {-2f}};
    private static final float[][] GRAD_OUTPUT = {{1f}, {-2f}, {3f}, {0f}, {2f}};

    private ComputeBackend host;

    @Before
    public void setUpGraph() {
        ComputationGraph.current().clear();
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDownGraph() {
        ComputationGraph.current().clear();
        TrainGpu.release(host);
    }

    @Test
    public void testConstructionWithValidParams() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 2);
        
        assertEquals(2, pp.numStages());
        assertEquals(2, pp.numMicroBatches());
        assertNotNull(pp.stages());
        assertEquals(2, pp.stages().size());
        
        pp.close();
    }

    @Test
    public void testStageAccess() {
        List<Module> stages = new ArrayList<>();
        MockModule stage0 = new MockModule();
        MockModule stage1 = new MockModule();
        stages.add(stage0);
        stages.add(stage1);
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        
        assertEquals(stage0, pp.stage(0));
        assertEquals(stage1, pp.stage(1));
        
        pp.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullStagesThrows() {
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        new PipelineParallel(null, backends, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEmptyStagesThrows() {
        List<Module> stages = new ArrayList<>();
        List<ComputeBackend> backends = new ArrayList<>();
        new PipelineParallel(stages, backends, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullBackendsThrows() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        new PipelineParallel(stages, null, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMismatchedStageAndBackendCountThrows() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        new PipelineParallel(stages, backends, 1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidMicroBatchCountZeroThrows() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        new PipelineParallel(stages, backends, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidMicroBatchCountNegativeThrows() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        new PipelineParallel(stages, backends, -1);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testStageIndexOutOfBoundsNegative() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        pp.stage(-1);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testStageIndexOutOfBoundsTooHigh() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        pp.stage(1);
    }

    @Test
    public void testTrainEvalModes() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        
        pp.train();
        pp.eval();
        
        pp.close();
    }

    @Test
    public void testZeroGrad() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        
        pp.zeroGrad();
        
        pp.close();
    }

    @Test
    public void testOperationsAfterCloseThrow() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        pp.close();
        
        try {
            pp.numStages();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
        
        try {
            pp.stages();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException expected) {}
    }

    @Test
    public void testCloseIsIdempotent() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        
        pp.close();
        pp.close();
        pp.close();
    }

    @Test
    public void testParameterCount() {
        List<Module> stages = new ArrayList<>();
        stages.add(new MockModule());
        stages.add(new MockModule());
        
        List<ComputeBackend> backends = new ArrayList<>();
        backends.add(null);
        backends.add(null);
        
        PipelineParallel pp = new PipelineParallel(stages, backends, 1);
        
        long count = pp.parameterCount();
        assertTrue(count >= 0);
        
        pp.close();
    }

    // ------------------------------------------------------------------ autograd

    /**
     * forward records (result is a leaf that requires grad) and backward with a non-uniform gradient, sliced per
     * micro-batch, gives every stage's parameter gradient and the input gradient of the unsplit computation
     * {@code y = x W0 W1}: {@code dW1 = (x W0)^T g}, {@code dW0 = x^T (g W1^T)}, {@code dx = g W1^T W0^T}. The caller's
     * graph is back at its size before the step.
     */
    @Test
    public void testBackwardMatchesTheUnsplitGradients() {
        Probe stage0 = new Probe(host, W0);
        Probe stage1 = new Probe(host, W1);
        PipelineParallel pp = pipeline(stage0, stage1, 3);
        int graphBefore = ComputationGraph.current().size();
        try (Tensor base = Tensor.fromArray(host, INPUT);
             Tensor x = base.withRequiresGrad(true);
             Tensor g = Tensor.fromArray(host, GRAD_OUTPUT)) {
            Tensor y = pp.forward(x);
            try {
                assertTrue("recorded result requires grad", y.requiresGrad());
                assertNull("recorded result is a leaf", y.gradFn());
                assertEquals(Shape.of(5, 1), y.shape());
                assertArrayEquals(flat(mm(mm(INPUT, W0), W1)), y.toFloatArray(), 0.0f);
                try (Tensor dx = pp.backward(g)) {
                    assertNotNull("input gradient", dx);
                    assertEquals(Shape.of(5, 2), dx.shape());
                    assertArrayEquals("dx", flat(mm(mm(GRAD_OUTPUT, transpose(W1)), transpose(W0))),
                        dx.toFloatArray(), 0.0f);
                }
                assertArrayEquals("dW0", flat(mm(transpose(INPUT), mm(GRAD_OUTPUT, transpose(W1)))),
                    stage0.weight.grad().toFloatArray(), 0.0f);
                assertArrayEquals("dW1", flat(mm(transpose(mm(INPUT, W0)), GRAD_OUTPUT)),
                    stage1.weight.grad().toFloatArray(), 0.0f);
                assertNull("the caller's input slot is not written", x.grad());
            } finally {
                y.close();
            }
            assertEquals("caller graph size", graphBefore, ComputationGraph.current().size());
        } finally {
            pp.close();
            stage0.close();
            stage1.close();
        }
    }

    /**
     * The caller's loss records against the result leaf; its gradient ({@code y.grad()}) drives the pipeline's
     * backward pass, here {@code d(sum y)/dW1 = column sums of x W0}.
     */
    @Test
    public void testCallerLossRoutesThroughTheResultLeaf() {
        Probe stage0 = new Probe(host, W0);
        Probe stage1 = new Probe(host, W1);
        PipelineParallel pp = pipeline(stage0, stage1, 2);
        int graphBefore = ComputationGraph.current().size();
        try (Tensor x = Tensor.fromArray(host, INPUT)) {
            Tensor y = pp.forward(x);
            try {
                Tensor loss = TensorOps.sum(y);
                loss.backward();
                loss.close();
                Tensor gy = y.grad();
                assertNotNull("the loss reached the result leaf", gy);
                assertNull("no input gradient when the input does not require grad", pp.backward(gy));
                float[][] ones = {{1f}, {1f}, {1f}, {1f}, {1f}};
                assertArrayEquals("dW1", flat(mm(transpose(mm(INPUT, W0)), ones)),
                    stage1.weight.grad().toFloatArray(), 0.0f);
                y.setGrad(null);
                gy.close();
            } finally {
                y.close();
            }
            assertEquals("caller graph size", graphBefore, ComputationGraph.current().size());
        } finally {
            pp.close();
            stage0.close();
            stage1.close();
        }
    }

    /** In evaluation mode, or with grad mode off, forward does not record and backward has nothing to run. */
    @Test
    public void testEvalAndNoGradDoNotRecord() {
        Probe stage0 = new Probe(host, W0);
        PipelineParallel pp = pipeline(stage0, new Probe(host, W1), 2);
        try (Tensor base = Tensor.fromArray(host, INPUT);
             Tensor x = base.withRequiresGrad(true);
             Tensor g = Tensor.fromArray(host, GRAD_OUTPUT)) {
            pp.eval();
            try (Tensor y = pp.forward(x)) {
                assertFalse("eval result does not require grad", y.requiresGrad());
            }
            assertBackwardRefused(pp, g);
            pp.train();
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor y = pp.forward(x)) {
                assertFalse("no-grad result does not require grad", y.requiresGrad());
            }
            assertBackwardRefused(pp, g);
            assertEquals("no stage graph was kept", Arrays.asList(0, 0, 0, 0), stage0.entrySizes());
            assertNull("no parameter gradient", stage0.weight.grad());
        } finally {
            pp.close();
        }
    }

    /** A recorded pass is backpropagated once; a second backward is refused. */
    @Test
    public void testBackwardTwiceIsRefused() {
        PipelineParallel pp = pipeline(new Probe(host, W0), new Probe(host, W1), 2);
        try (Tensor x = Tensor.fromArray(host, INPUT);
             Tensor g = Tensor.fromArray(host, GRAD_OUTPUT)) {
            pp.forward(x).close();
            assertNull(pp.backward(g));
            assertBackwardRefused(pp, g);
        } finally {
            pp.close();
        }
    }

    /**
     * Release points on the stage threads, traced through the stage graph's size when each micro-batch enters stage 0:
     * micro-batch 0 of every pass starts on an empty graph, both after a backward pass and when the previous recorded
     * pass was never backpropagated (released by the next forward); within a pass the earlier micro-batches' graphs
     * are retained.
     */
    @Test
    public void testRetainedGraphsAreReleasedOnBackwardAndOnTheNextForward() {
        Probe stage0 = new Probe(host, W0);
        PipelineParallel pp = pipeline(stage0, new Probe(host, W1), 2);
        try (Tensor x = Tensor.fromArray(host, INPUT);
             Tensor g = Tensor.fromArray(host, GRAD_OUTPUT)) {
            pp.forward(x).close();
            pp.forward(x).close();
            assertNull(pp.backward(g));
            pp.forward(x).close();
            List<Integer> sizes = stage0.entrySizes();
            assertEquals(6, sizes.size());
            for (int pass = 0; pass < 3; pass++) {
                assertEquals("pass " + pass + " starts on an empty stage graph", 0, (int) sizes.get(2 * pass));
                assertTrue("pass " + pass + " retains micro-batch 0", sizes.get(2 * pass + 1) > 0);
            }
        } finally {
            pp.close();
        }
    }

    /**
     * A stage failing during a recorded pass: the failure is rethrown, the graphs that pass retained are released,
     * backward has nothing to run, and the next pass records and backpropagates normally.
     */
    @Test
    public void testRecordedStageFailureReleasesAndThePipelineStaysUsable() {
        Probe stage0 = new Probe(host, W0);
        FailOnSecondCall failing = new FailOnSecondCall();
        Probe stage2 = new Probe(host, W1);
        PipelineParallel pp = new PipelineParallel(Arrays.asList(stage0, failing, stage2),
            Arrays.asList(host, host, host), 3);
        try (Tensor x = Tensor.fromArray(host, INPUT);
             Tensor g = Tensor.fromArray(host, GRAD_OUTPUT)) {
            try {
                pp.forward(x).close();
                fail("the stage failure was not rethrown");
            } catch (IllegalStateException e) {
                assertEquals(FailOnSecondCall.MESSAGE, e.getMessage());
            }
            assertBackwardRefused(pp, g);
            pp.forward(x).close();
            assertNull(pp.backward(g));
            assertArrayEquals("dW1 after recovery", flat(mm(transpose(mm(INPUT, W0)), GRAD_OUTPUT)),
                stage2.weight.grad().toFloatArray(), 0.0f);
            List<Integer> sizes = stage0.entrySizes();
            assertEquals("the pass after the failure starts on an empty stage graph", 0,
                (int) sizes.get(sizes.size() - 3));
        } finally {
            pp.close();
        }
    }

    /** A gradient of the wrong shape is refused and the recorded pass is kept for a correct one. */
    @Test
    public void testWrongGradientShapeIsRefusedAndTheRecordingKept() {
        Probe stage1 = new Probe(host, W1);
        PipelineParallel pp = pipeline(new Probe(host, W0), stage1, 2);
        try (Tensor x = Tensor.fromArray(host, INPUT);
             Tensor wrong = Tensor.zeros(host, Shape.of(4, 1));
             Tensor g = Tensor.fromArray(host, GRAD_OUTPUT)) {
            pp.forward(x).close();
            try {
                pp.backward(wrong);
                fail("a gradient of the wrong shape was accepted");
            } catch (IllegalArgumentException expected) {
            }
            assertNull(pp.backward(g));
            assertNotNull(stage1.weight.grad());
        } finally {
            pp.close();
        }
    }

    /**
     * Auxiliary loss terms recorded on a stage thread are backpropagated by {@link PipelineParallel#backward}: the last
     * stage leaves {@code sum(W1)} per micro-batch, so with two micro-batches {@code dW1} is the unsplit gradient plus
     * 2 in every element, and {@code dW0} and {@code dx} are unchanged (the term does not depend on them).
     */
    @Test
    public void testStageAuxiliaryTermsAreBackpropagated() {
        Probe stage0 = new Probe(host, W0);
        AuxiliaryProbe stage1 = new AuxiliaryProbe(host, W1);
        PipelineParallel pp = pipeline(stage0, stage1, 2);
        try (Tensor base = Tensor.fromArray(host, INPUT);
             Tensor x = base.withRequiresGrad(true);
             Tensor g = Tensor.fromArray(host, GRAD_OUTPUT)) {
            Tensor y = pp.forward(x);
            try (Tensor dx = pp.backward(g)) {
                assertArrayEquals("dx", flat(mm(mm(GRAD_OUTPUT, transpose(W1)), transpose(W0))), dx.toFloatArray(),
                    0.0f);
            } finally {
                y.close();
            }
            float[] expected = flat(mm(transpose(mm(INPUT, W0)), GRAD_OUTPUT));
            for (int i = 0; i < expected.length; i++) {
                expected[i] += 2f;
            }
            assertArrayEquals("dW1", expected, stage1.weight.grad().toFloatArray(), 0.0f);
            assertArrayEquals("dW0", flat(mm(transpose(INPUT), mm(GRAD_OUTPUT, transpose(W1)))),
                stage0.weight.grad().toFloatArray(), 0.0f);
        } finally {
            pp.close();
            stage0.close();
            stage1.close();
        }
    }

    private PipelineParallel pipeline(Module first, Module second, int microBatches) {
        return new PipelineParallel(Arrays.asList(first, second), Arrays.asList(host, host), microBatches);
    }

    private static void assertBackwardRefused(PipelineParallel pp, Tensor g) {
        try {
            pp.backward(g);
            fail("backward ran without a recorded forward pass");
        } catch (IllegalStateException expected) {
        }
    }

    private static float[][] mm(float[][] a, float[][] b) {
        float[][] out = new float[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                float sum = 0f;
                for (int k = 0; k < b.length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                out[i][j] = sum;
            }
        }
        return out;
    }

    private static float[][] transpose(float[][] a) {
        float[][] out = new float[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                out[j][i] = a[i][j];
            }
        }
        return out;
    }

    private static float[] flat(float[][] a) {
        float[] out = new float[a.length * a[0].length];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, out, i * a[0].length, a[0].length);
        }
        return out;
    }

    /**
     * {@code x W} recorded by TensorOps with one parameter; records, per call, the size of the calling thread's graph
     * when the call starts.
     */
    static final class Probe extends Module {
        final Parameter weight;
        private final List<Integer> entrySizes = Collections.synchronizedList(new ArrayList<>());

        Probe(ComputeBackend b, float[][] w) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, w), "weight"));
        }

        List<Integer> entrySizes() {
            return new ArrayList<>(entrySizes);
        }

        @Override
        public Tensor forward(Tensor input) {
            entrySizes.add(ComputationGraph.current().size());
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    /** {@code x W} recorded by TensorOps; each forward pass leaves the auxiliary term {@code sum(W)}. */
    static final class AuxiliaryProbe extends Module {
        final Parameter weight;

        AuxiliaryProbe(ComputeBackend b, float[][] w) {
            weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, w), "weight"));
        }

        @Override
        public Tensor forward(Tensor input) {
            setAuxiliaryLoss(TensorOps.sum(weight.tensor()));
            return TensorOps.matmul(input, weight.tensor());
        }
    }

    /** The identity on every call except the second, which throws {@link #MESSAGE}. */
    static final class FailOnSecondCall extends Module {
        static final String MESSAGE = "stage failed on purpose";
        private int calls;

        @Override
        public Tensor forward(Tensor input) {
            if (++calls == 2) {
                throw new IllegalStateException(MESSAGE);
            }
            return input;
        }
    }

    static class MockModule extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }
}
