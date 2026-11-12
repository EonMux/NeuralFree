package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * How a first-order backward pass frees the graph and accumulates gradients: a node is released as soon as its
 * gradient function ran (saved tensors dropped, output closed), a failed pass has released exactly the nodes that ran,
 * a gradient held exclusively by its slot takes the next contribution in place while one shared with another handle
 * is replaced, and {@link ComputationGraph#reachedByLastBackward} reports the leaves a pass delivered to, including
 * the leaves reached inside checkpoint recomputes. An output that a custom gradient function saves directly, without
 * consuming it, stays open until that function ran even when its own node is released first. Runs on
 * {@link TrainParity#hostBackend()}.
 */
public class BackwardReleaseTest {

    private ComputeBackend backend;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        backend = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        backend.close();
    }

    @Test
    public void eachNodeIsReleasedAsSoonAsItsGradientFunctionRan() {
        try (Tensor x = leaf(1f, 2f, 3f)) {
            List<String> seen = new ArrayList<>();
            Tensor a = TensorOps.mul(x, 2f);
            Tensor[] later = new Tensor[1];
            Tensor b = observed(a, () -> {
                ComputationGraph graph = ComputationGraph.current();
                seen.add("consumer node in graph: " + (graph.getNode(later[0]) != null));
                seen.add("consumer output closed: " + isClosed(later[0]));
            });
            Tensor c = TensorOps.mul(b, 3f);
            later[0] = c;
            Tensor loss = TensorOps.sum(c);
            int recorded = ComputationGraph.current().size();
            assertEquals(4, recorded);

            loss.backward();

            assertEquals(Arrays.asList("consumer node in graph: false", "consumer output closed: true"), seen);
            assertEquals("the whole traversed graph is released", 0, ComputationGraph.current().size());
            assertArrayEquals(new float[]{6f, 6f, 6f}, x.grad().toFloatArray(), 0f);
            assertFalse("the root stays the caller's", isClosed(loss));
            loss.close();
        }
    }

    /**
     * A failing gradient function ends the pass, and the pass still releases the whole traversed graph: the nodes that
     * ran were released as they ran, so leaving the rest would strand nodes no root reaches any more.
     */
    @Test
    public void aFailedPassReleasesTheTraversedGraph() {
        try (Tensor x = leaf(1f, 2f)) {
            Tensor a = TensorOps.mul(x, 2f);
            Tensor b = observed(a, () -> {
                throw new IllegalStateException("gradient function fails");
            });
            Tensor c = TensorOps.mul(b, 3f);
            Tensor loss = TensorOps.sum(c);
            ComputationGraph graph = ComputationGraph.current();
            try {
                loss.backward();
                fail("the failing gradient function must fail the pass");
            } catch (RuntimeException expected) {
                assertTrue(String.valueOf(expected.getCause()), expected.getCause() instanceof IllegalStateException);
            }
            assertNull("the root node ran and is released", graph.getNode(loss));
            assertNull("the node after it ran and is released", graph.getNode(c));
            assertTrue("its output is closed", isClosed(c));
            assertNull("the failing node is released", graph.getNode(b));
            assertNull("the node that never ran is released", graph.getNode(a));
            assertTrue("their outputs are closed", isClosed(b) && isClosed(a));
            assertEquals(0, graph.size());
            assertNull("no gradient reached the leaf", x.grad());
            try {
                loss.backward();
                fail("a released root cannot be backpropagated again");
            } catch (IllegalStateException expected) {
                assertEquals(ComputationGraph.RELEASED_GRAPH_MESSAGE, expected.getMessage());
            }
            loss.close();
        }
    }

    /**
     * {@code loss = sum(f + p)} with {@code p = 2x} and {@code f} a custom node over {@code q = 3x} that saves the
     * output object {@code p} itself (not a detached handle) and is not its consumer. The pass visits {@code p}'s node
     * before {@code f}'s, so releasing {@code p} as soon as its node ran would close the tensor {@code f}'s backward
     * reads.
     */
    @Test
    public void anOutputACustomFunctionSavesDirectlyOutlivesItsOwnNode() {
        try (Tensor x = leaf(1f, 2f)) {
            Tensor p = TensorOps.mul(x, 2f);
            Tensor q = TensorOps.mul(x, 3f);
            Tensor raw = Autograd.untracked(() -> TensorOps.mul(q, p));
            Tensor f = Autograd.record(raw, Collections.singletonList(q), out -> new GradientFunction() {
                private List<Tensor> saved = Collections.singletonList(p);

                @Override
                public List<Tensor> backward(Tensor gradOutput) {
                    return Collections.singletonList(TensorOps.mul(gradOutput, saved.get(0)));
                }

                @Override
                public List<Tensor> savedTensors() {
                    return saved;
                }

                @Override
                public void clearSavedTensors() {
                    saved = Collections.emptyList();
                }
            });
            Tensor loss = TensorOps.sum(TensorOps.add(f, p));

            loss.backward();

            assertArrayEquals("3 * p + 2", new float[]{8f, 14f}, x.grad().toFloatArray(), 0f);
            assertEquals(0, ComputationGraph.current().size());
            assertTrue("the saved output is closed with the rest of the pass", isClosed(p));
            loss.close();
        }
    }

    @Test
    public void aRetainedGraphIsKeptWholeForASecondPass() {
        try (Tensor x = leaf(1f, 2f)) {
            Tensor c = TensorOps.mul(TensorOps.mul(x, 2f), 3f);
            Tensor loss = TensorOps.sum(c);
            ComputationGraph.current().backward(loss, null, true);
            assertEquals(3, ComputationGraph.current().size());
            assertFalse(isClosed(c));
            loss.backward();
            assertEquals(0, ComputationGraph.current().size());
            assertArrayEquals(new float[]{12f, 12f}, x.grad().toFloatArray(), 0f);
            loss.close();
        }
    }

    @Test
    public void anExclusivelyHeldGradientTakesTheNextContributionInPlace() {
        try (Tensor x = leaf(1f, -2f, 0.5f)) {
            backwardOf(TensorOps.sum(TensorOps.mul(x, x)));
            Tensor first = x.grad();
            float[] once = first.toFloatArray();
            Tensor twice = TensorOps.add(TensorOps.mul(x, 3f), TensorOps.mul(x, 4f));
            backwardOf(TensorOps.sum(twice));
            twice.close();

            assertSame("the slot keeps its tensor", first, x.grad());
            float[] expected = new float[once.length];
            for (int i = 0; i < once.length; i++) {
                expected[i] = (once[i] + 3f) + 4f;
            }
            assertArrayEquals(expected, x.grad().toFloatArray(), 0f);
            assertTrue(ComputationGraph.current().reachedByLastBackward(x));
        }
    }

    @Test
    public void aGradientSharedWithAnotherHandleIsReplacedNotWritten() {
        try (Tensor x = leaf(1f, -2f, 0.5f)) {
            backwardOf(TensorOps.sum(TensorOps.mul(x, x)));
            Tensor first = x.grad();
            try (Tensor kept = first.detach()) {
                float[] before = kept.toFloatArray();
                backwardOf(TensorOps.sum(TensorOps.mul(x, 5f)));

                assertNotSame("the shared gradient is replaced", first, x.grad());
                assertArrayEquals("the other handle still reads the old gradient", before, kept.toFloatArray(), 0f);
                float[] expected = new float[before.length];
                for (int i = 0; i < before.length; i++) {
                    expected[i] = before[i] + 5f;
                }
                assertArrayEquals(expected, x.grad().toFloatArray(), 0f);
            }
        }
    }

    @Test
    public void reachedByLastBackwardNamesTheLeavesOfTheLastPassOnly() {
        try (Parameter used = new Parameter(Tensor.fromArray(backend, new float[]{1f, 2f}), "used");
             Parameter idle = new Parameter(Tensor.fromArray(backend, new float[]{3f, 4f}), "idle")) {
            backwardOf(TensorOps.sum(TensorOps.mul(TensorOps.add(used.tensor(), idle.tensor()), 2f)));
            assertTrue(used.reachedByLastBackward());
            assertTrue(idle.reachedByLastBackward());

            backwardOf(TensorOps.sum(TensorOps.mul(used.tensor(), 2f)));
            assertTrue("accumulated in place, still reached", used.reachedByLastBackward());
            assertFalse("holds a gradient from an earlier pass only", idle.reachedByLastBackward());
            assertNotNull(idle.grad());
        }
    }

    @Test
    public void leavesReachedInsideEveryCheckpointRecomputeCount() {
        try (Tensor x = leaf(1f, 2f);
             Tensor w1 = leaf(0.5f, 1.5f);
             Tensor w2 = leaf(2f, -1f)) {
            Tensor h = CheckpointFunction.checkpoint(t -> TensorOps.mul(t, w1), x);
            Tensor y = CheckpointFunction.checkpoint(t -> TensorOps.mul(t, w2), h);
            backwardOf(TensorOps.sum(y));
            h.close();
            y.close();

            ComputationGraph graph = ComputationGraph.current();
            assertTrue("the first segment's weight", graph.reachedByLastBackward(w1));
            assertTrue("the second segment's weight", graph.reachedByLastBackward(w2));
            assertArrayEquals(new float[]{2f, -2f}, w1.grad().toFloatArray(), 0f);
            assertArrayEquals(new float[]{0.5f, 3f}, w2.grad().toFloatArray(), 0f);
        }
    }

    private Tensor leaf(float... values) {
        Tensor raw = Tensor.fromArray(backend, values);
        Tensor leaf = raw.withRequiresGrad(true);
        if (leaf != raw) {
            raw.close();
        }
        return leaf;
    }

    private static void backwardOf(Tensor loss) {
        loss.backward();
        loss.close();
    }

    /** {@code input} unchanged, recorded with a gradient function that runs {@code hook} and passes the gradient on. */
    private static Tensor observed(Tensor input, Runnable hook) {
        Tensor raw = Autograd.untracked(() -> TensorOps.mul(input, 1f));
        return Autograd.record(raw, Collections.singletonList(input), out -> new GradientFunction() {
            @Override
            public List<Tensor> backward(Tensor gradOutput) {
                hook.run();
                return Collections.singletonList(gradOutput.detach());
            }

            @Override
            public List<Tensor> savedTensors() {
                return Collections.emptyList();
            }
        });
    }

    private static boolean isClosed(Tensor t) {
        try {
            t.shape();
            t.toFloatArray();
            return false;
        } catch (IllegalStateException closed) {
            return true;
        }
    }
}
