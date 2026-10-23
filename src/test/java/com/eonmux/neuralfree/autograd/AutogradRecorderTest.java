package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
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
 * The recorder ({@link Autograd}) and the graph semantics of {@link ComputationGraph}: the recording rule, backward
 * without recording, leaf backward and released non-leaves, {@link Autograd#grad} with its reachability mask and
 * create-graph mode, the create-graph lifetime rule and {@link ComputationGraph.StepScope}. Runs on
 * {@link TrainParity#hostBackend()}. The recorded operations are defined here with the facade pattern of
 * {@link Autograd}, so the expectations depend only on the recorder and the graph.
 */
public class AutogradRecorderTest {

    private ComputeBackend backend;
    private ComputationGraph graph;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        backend = TrainParity.hostBackend();
        graph = ComputationGraph.current();
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        backend.close();
    }

    // ------------------------------------------------------------------ recording rule

    @Test
    public void needsGradTruthTable() {
        try (Tensor leaf = leaf(1f, 2f); Tensor constant = constant(3f, 4f)) {
            assertTrue(Autograd.needsGrad(leaf));
            assertFalse(Autograd.needsGrad(constant));
            assertFalse(Autograd.needsGrad((Tensor) null));
            assertTrue(Autograd.needsGrad(constant, leaf));
            assertFalse(Autograd.needsGrad(constant, null));
            assertTrue(Autograd.needsGrad(constant, constant, leaf));
            assertFalse(Autograd.needsGrad(constant, null, constant));
            assertTrue(Autograd.needsGrad(Arrays.asList(constant, null, leaf)));
            assertFalse(Autograd.needsGrad(Arrays.asList(constant, null)));
            assertFalse(Autograd.needsGrad((List<Tensor>) null));
            try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
                assertFalse(Autograd.needsGrad(leaf));
                assertFalse(Autograd.needsGrad(constant, leaf));
                assertFalse(Autograd.needsGrad(Collections.singletonList(leaf)));
                try (ComputationGraph.EnableGradScope on = ComputationGraph.enableGrad()) {
                    assertTrue(Autograd.needsGrad(leaf));
                }
                assertFalse("enableGrad restores the previous mode", ComputationGraph.isGradEnabled());
            }
            assertTrue(ComputationGraph.isGradEnabled());
        }
    }

    @Test
    public void noGradResultsDoNotRequireGrad() {
        try (Tensor x = leaf(1f, 2f, 3f); Tensor w = leaf(4f, 5f, 6f)) {
            Tensor product;
            try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
                product = mul(x, w);
            }
            try (Tensor r = product) {
                assertFalse(r.requiresGrad());
                assertNull(r.gradFn());
            }
            assertEquals(0, graph.size());
        }
    }

    @Test
    public void recordedResultIsANonLeafDistinctFromItsInput() {
        try (Tensor x = leaf(1f, 2f)) {
            Tensor y = Autograd.record(x, Collections.singletonList(x), out -> new PassFn());
            assertNotSame("a raw result identical to an input gets its own handle", x, y);
            assertTrue(y.requiresGrad());
            assertNotNull(y.gradFn());
            assertNotNull(graph.getNode(y));
            assertArrayEquals(new float[]{1f, 2f}, x.toFloatArray(), 0f);
            graph.release(y);
            y.close();
        }
    }

    @Test
    public void recordFailureClosesTheOutput() {
        try (Tensor x = leaf(1f, 2f); Tensor w = leaf(3f, 4f)) {
            Tensor raw = Autograd.untracked(() -> TensorOps.mul(x, w));
            Tensor[] marked = new Tensor[1];
            try {
                Autograd.record(raw, Arrays.asList(x, w), out -> {
                    marked[0] = out;
                    throw new IllegalStateException("factory failure");
                });
                fail("record should rethrow the factory failure");
            } catch (IllegalStateException expected) {
                assertEquals("factory failure", expected.getMessage());
            }
            assertClosed(marked[0], "the marked output");
            assertEquals(0, graph.size());
        }
    }

    @Test
    public void recordOutsideGradModeThrowsAndClosesTheRaw() {
        try (Tensor x = leaf(1f, 2f)) {
            Tensor raw = Autograd.untracked(() -> TensorOps.mul(x, x));
            try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
                Autograd.record(raw, Collections.singletonList(x), out -> new PassFn());
                fail("record requires needsGrad");
            } catch (IllegalStateException expected) {
                assertClosed(raw, "the raw result");
            }
            assertEquals(0, graph.size());
        }
    }

    @Test
    public void recordEachIsAllOrNothing() {
        try (Tensor x = leaf(1f, 2f)) {
            Tensor first = Autograd.untracked(() -> TensorOps.mul(x, x));
            Tensor second = Autograd.untracked(() -> TensorOps.mul(x, x));
            try {
                Autograd.recordEach(Arrays.asList(first, second), Collections.singletonList(x), i -> {
                    if (i == 1) {
                        throw new IllegalStateException("second part fails");
                    }
                    return new PassFn();
                });
                fail("recordEach should rethrow");
            } catch (IllegalStateException expected) {
                assertEquals(0, graph.size());
            }
            List<Tensor> parts = Autograd.recordEach(Arrays.asList(
                Autograd.untracked(() -> TensorOps.mul(x, x)), Autograd.untracked(() -> TensorOps.mul(x, x))),
                Collections.singletonList(x), i -> new PassFn());
            assertEquals(2, graph.size());
            for (Tensor part : parts) {
                assertNotNull(graph.getNode(part));
                graph.release(part);
                part.close();
            }
            assertEquals(0, graph.size());
        }
    }

    @Test
    public void reattachOutsideACreateGraphPassReturnsTheHandle() {
        try (Tensor x = leaf(1f, 2f); Tensor handle = x.detach()) {
            assertFalse(Autograd.isCreatingGraph());
            assertSame(handle, Autograd.reattach(handle, x));
            assertEquals(0, graph.size());
        }
    }

    // ------------------------------------------------------------------ backward

    @Test
    public void backwardRecordsNothing() {
        try (Tensor x = leaf(1f, 2f, 3f); Tensor w = leaf(4f, 5f, 6f)) {
            Tensor y = sum(mul(x, w));
            int recorded = graph.size();
            graph.backward(y, null, true);
            assertEquals("a first-order pass records no node", recorded, graph.size());
            assertArrayEquals(new float[]{4f, 5f, 6f}, x.grad().toFloatArray(), 0f);
            assertArrayEquals(new float[]{1f, 2f, 3f}, w.grad().toFloatArray(), 0f);
            assertFalse(x.grad().requiresGrad());
            assertNull(x.grad().gradFn());
            graph.release(y);
            y.close();
            graph.zeroGrad();
        }
    }

    @Test
    public void leafBackwardFillsTheSlot() {
        try (Tensor x = leaf(1f, 2f); Tensor seed = constant(0.5f, -2f)) {
            graph.backward(x, seed);
            assertNotSame("the slot holds its own reference", seed, x.grad());
            assertArrayEquals(new float[]{0.5f, -2f}, x.grad().toFloatArray(), 0f);
            graph.backward(x, seed);
            assertArrayEquals("a second leaf backward accumulates", new float[]{1f, -4f},
                x.grad().toFloatArray(), 0f);
            assertArrayEquals("the seed stays the caller's", new float[]{0.5f, -2f}, seed.toFloatArray(), 0f);
            graph.zeroGrad();
        }
    }

    @Test
    public void secondBackwardWithoutRetainThrows() {
        try (Tensor x = leaf(1f, 2f); Tensor w = leaf(3f, 4f)) {
            Tensor y = sum(mul(x, w));
            graph.backward(y);
            float[] first = x.grad().toFloatArray();
            try {
                graph.backward(y);
                fail("the graph of y was released");
            } catch (IllegalStateException expected) {
                assertTrue(expected.getMessage().contains("released"));
            }
            assertArrayEquals("nothing was seeded", first, x.grad().toFloatArray(), 0f);
            assertNull("y is no leaf: its own slot stays empty", y.grad());
            y.close();
            graph.zeroGrad();
        }
    }

    @Test
    public void backwardIntoAReleasedIntermediateThrows() {
        try (Tensor x = leaf(1f, 2f); Tensor w = leaf(3f, 4f)) {
            Tensor h = mul(x, w);
            graph.release(h);
            Tensor y = sum(h);
            try {
                graph.backward(y);
                fail("h is a released non-leaf");
            } catch (RuntimeException expected) {
                assertTrue(causedBy(expected, IllegalStateException.class));
            }
            assertNull("no slot was written", x.grad());
            assertNull(w.grad());
            assertNull(h.grad());
            graph.release(y);
            y.close();
            assertEquals(0, graph.size());

            try (Tensor copy = h.detach(); Tensor fresh = copy.withRequiresGrad(true)) {
                Tensor z = sum(mul(fresh, w));
                graph.backward(z);
                assertArrayEquals("a detached copy is a new leaf", new float[]{3f, 4f},
                    fresh.grad().toFloatArray(), 0f);
                assertArrayEquals(new float[]{3f, 8f}, w.grad().toFloatArray(), 0f);
                z.close();
            }
            h.close();
            graph.zeroGrad();
        }
    }

    // ------------------------------------------------------------------ Autograd.grad

    @Test
    public void gradLeavesEverySlotUntouched() {
        try (Tensor x = leaf(1f, 2f, 3f); Tensor w = leaf(4f, 5f, 6f)) {
            Tensor h = mul(x, w);
            Tensor y = sum(h);
            List<Tensor> grads = Autograd.grad(y, Arrays.asList(h, x), null, true, false);
            assertEquals(2, grads.size());
            assertArrayEquals("node-output target", new float[]{1f, 1f, 1f}, grads.get(0).toFloatArray(), 0f);
            assertArrayEquals("leaf target upstream of it", new float[]{4f, 5f, 6f}, grads.get(1).toFloatArray(),
                0f);
            assertNull(x.grad());
            assertNull(w.grad());
            assertNull(h.grad());
            assertEquals("retained", 2, graph.size());
            for (Tensor g : grads) {
                g.close();
            }
            List<Tensor> again = Autograd.grad(y, Collections.singletonList(w), null, false, false);
            assertArrayEquals(new float[]{1f, 2f, 3f}, again.get(0).toFloatArray(), 0f);
            again.get(0).close();
            assertEquals("released without retainGraph", 0, graph.size());
            y.close();
        }
    }

    @Test
    public void nonRetainedGradKeepsTheGraphOfANodeOutputTarget() {
        try (Tensor x = leaf(1f, 2f, 3f); Tensor w = leaf(4f, 5f, 6f)) {
            Tensor h = mul(x, w);
            Tensor hw = mul(h, w);
            Tensor y = sum(hw);
            assertEquals(3, graph.size());
            Tensor g = Autograd.grad(y, Collections.singletonList(h), null, false, false).get(0);
            assertArrayEquals("dy/dh = w", new float[]{4f, 5f, 6f}, g.toFloatArray(), 0f);
            assertEquals("only the nodes downstream of h were released", 1, graph.size());
            assertClosed(hw, "an intermediate the pass ran through");
            assertArrayEquals("h stays open", new float[]{4f, 10f, 18f}, h.toFloatArray(), 0f);

            graph.backward(h, g);
            assertArrayEquals("g * w", new float[]{16f, 25f, 36f}, x.grad().toFloatArray(), 0f);
            assertArrayEquals("g * x", new float[]{4f, 10f, 18f}, w.grad().toFloatArray(), 0f);
            assertEquals(0, graph.size());
            g.close();
            h.close();
            y.close();
            graph.zeroGrad();
        }
    }

    @Test
    public void createGraphInsideNoGradReturnsGradientsThatRequireGrad() {
        try (Tensor x = leaf(1f, 2f, 3f); Tensor w = leaf(4f, 5f, 6f)) {
            Tensor y = sum(mul(x, w));
            List<Tensor> grads;
            try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
                grads = Autograd.grad(y, Collections.singletonList(x), null, false, true);
                assertFalse("the caller's mode is restored", ComputationGraph.isGradEnabled());
            }
            Tensor g = grads.get(0);
            assertTrue(g.requiresGrad());
            assertNotNull(g.gradFn());
            assertArrayEquals(new float[]{4f, 5f, 6f}, g.toFloatArray(), 0f);
            assertNull(x.grad());
            assertFalse(Autograd.isCreatingGraph());
            graph.release(y);
            graph.release(g);
            y.close();
            g.close();
            assertEquals(0, graph.size());
        }
    }

    @Test
    public void createGraphKeepsOnlyTheNodesItsResultsAreComputedFrom() {
        try (Tensor x = leaf(1f, 2f, 3f); Tensor w = leaf(4f, 5f, 6f)) {
            Tensor y = sum(mul(x, w));
            int forward = graph.size();
            Tensor g = Autograd.grad(y, Collections.singletonList(x), null, false, true).get(0);
            assertEquals("forward graph retained plus the product node of g and the identity node of w's saved "
                + "handle; the w-gradient branch is pruned", forward + 2, graph.size());
            assertNotNull(graph.getNode(g));

            Tensor s = sum(g);
            graph.backward(s);
            assertArrayEquals("second order reaches w through the reattached saved handle",
                new float[]{1f, 1f, 1f}, w.grad().toFloatArray(), 0f);
            assertNull(x.grad());
            assertEquals("the kept nodes are released by the next non-retained backward", forward, graph.size());

            graph.release(y);
            assertEquals(0, graph.size());
            s.close();
            g.close();
            y.close();
            graph.zeroGrad();
        }
    }

    @Test
    public void createGraphThrowsForUnsupportedNodesItWouldRun() {
        try (Tensor x = leaf(1f, 2f); Tensor w = leaf(3f, 4f)) {
            OpaqueFn opaque = new OpaqueFn();
            Tensor u = Autograd.record(Autograd.untracked(() -> TensorOps.mul(w, w)), Collections.singletonList(w),
                out -> opaque);
            Tensor y = sum(mul(x, u));
            int recorded = graph.size();
            try {
                Autograd.grad(y, Collections.singletonList(w), null, true, true);
                fail("the opaque node lies on the path to w");
            } catch (UnsupportedOperationException expected) {
                assertTrue(expected.getMessage().contains(OpaqueFn.class.getName()));
            }
            try {
                graph.backward(y, null, true, true);
                fail("backward with createGraph runs every traversed node");
            } catch (UnsupportedOperationException expected) {
                assertEquals("nothing was computed", 0, opaque.calls);
            }
            assertEquals(recorded, graph.size());
            assertNull(x.grad());
            assertNull(w.grad());

            Tensor g = Autograd.grad(y, Collections.singletonList(x), null, true, true).get(0);
            assertEquals("the opaque node is outside the mask", 0, opaque.calls);
            assertArrayEquals(new float[]{9f, 16f}, g.toFloatArray(), 0f);
            assertTrue(g.requiresGrad());
            graph.release(y);
            graph.release(g);
            g.close();
            y.close();
            u.close();
            assertEquals(0, graph.size());
        }
    }

    // ------------------------------------------------------------------ steps

    @Test
    public void stepScopeRestoresSizeAndClosesDiscardedOutputs() {
        try (Tensor x = leaf(1f, 2f); Tensor w = leaf(3f, 4f)) {
            Tensor before = mul(x, w);
            int start = graph.size();
            Tensor discarded;
            Tensor kept;
            try (ComputationGraph.StepScope step = graph.step()) {
                discarded = mul(x, w);
                kept = step.spare(sum(mul(x, before)));
            }
            assertEquals(start, graph.size());
            assertClosed(discarded, "an output the step discarded");
            assertEquals(1f * 3f + 2f * 8f, kept.toFloatArray()[0], 0f);
            assertArrayEquals("a node recorded before the step keeps its output", new float[]{3f, 8f},
                before.toFloatArray(), 0f);
            kept.close();
            graph.release(before);
            before.close();
            assertEquals(0, graph.size());
        }
    }

    @Test
    public void stepScopeRemovesWhatACreateGraphPassKept() {
        try (Tensor x = leaf(1f, 2f, 3f); Tensor w = leaf(4f, 5f, 6f)) {
            int start = graph.size();
            try (ComputationGraph.StepScope step = graph.step()) {
                Tensor y = sum(mul(x, w));
                Tensor g = Autograd.grad(y, Collections.singletonList(x), null, true, true).get(0);
                Tensor penalty = sum(mul(g, g));
                graph.backward(penalty);
                assertArrayEquals("d/dw of sum(w*w) through the gradient", new float[]{8f, 10f, 12f},
                    w.grad().toFloatArray(), 0f);
                penalty.close();
            }
            assertEquals(start, graph.size());
            graph.zeroGrad();
        }
    }

    // ------------------------------------------------------------------ recorded test operations

    /** {@code a * b} for tensors of one shape, recorded with the facade pattern. */
    static Tensor mul(Tensor a, Tensor b) {
        if (!Autograd.needsGrad(a, b)) {
            return TensorOps.mul(a, b);
        }
        Tensor raw = Autograd.untracked(() -> TensorOps.mul(a, b));
        return Autograd.record(raw, Arrays.asList(a, b), out -> {
            MulFn fn = new MulFn(a.requiresGrad(), b.requiresGrad());
            fn.save(a, b);
            return fn;
        });
    }

    /** The sum of all elements as a scalar, recorded with the facade pattern. */
    static Tensor sum(Tensor a) {
        if (!Autograd.needsGrad(a)) {
            return TensorOps.sum(a);
        }
        Tensor raw = Autograd.untracked(() -> TensorOps.sum(a));
        Shape shape = a.shape();
        return Autograd.record(raw, Collections.singletonList(a), out -> new SumFn(shape));
    }

    /** The scalar {@code s} broadcast to {@code shape}, recorded with the facade pattern. */
    static Tensor broadcast(Tensor s, Shape shape) {
        if (!Autograd.needsGrad(s)) {
            return broadcastRaw(s, shape);
        }
        Tensor raw = Autograd.untracked(() -> broadcastRaw(s, shape));
        return Autograd.record(raw, Collections.singletonList(s), out -> new BroadcastFn());
    }

    private static Tensor broadcastRaw(Tensor s, Shape shape) {
        try (Tensor ones = Tensor.ones(s.backend(), shape)) {
            return TensorOps.mul(ones, s);
        }
    }

    /** {@code a * b}: {@code ga = gy * b}, {@code gb = gy * a}; saves a and b. */
    static final class MulFn extends GradFunctions.SavingGradFn {
        private final boolean needA;
        private final boolean needB;

        MulFn(boolean needA, boolean needB) {
            this.needA = needA;
            this.needB = needB;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor ga = needA ? mul(gradOutput, saved(1)) : null;
            try {
                return Arrays.asList(ga, needB ? mul(gradOutput, saved(0)) : null);
            } catch (RuntimeException e) {
                if (ga != null) {
                    ga.close();
                }
                throw e;
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /** {@code sum(a)}: {@code ga = broadcast(gy, a.shape)}. */
    static final class SumFn implements GradientFunction {
        private final Shape inputShape;

        SumFn(Shape inputShape) {
            this.inputShape = inputShape;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(broadcast(gradOutput, inputShape));
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /** {@code broadcast(s, shape)}: {@code gs = sum(gy)}. */
    static final class BroadcastFn implements GradientFunction {
        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(sum(gradOutput));
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /** Passes the gradient through; supports create-graph. */
    static final class PassFn implements GradientFunction {
        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(gradOutput);
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /** A backward that cannot record itself; counts its calls. */
    static final class OpaqueFn implements GradientFunction {
        private int calls;

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            calls++;
            return Collections.singletonList(gradOutput);
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }
    }

    // ------------------------------------------------------------------ helpers

    private Tensor leaf(float... values) {
        try (Tensor t = Tensor.fromArray(backend, values)) {
            return t.withRequiresGrad(true);
        }
    }

    private Tensor constant(float... values) {
        return Tensor.fromArray(backend, values);
    }

    static void assertClosed(Tensor t, String what) {
        try {
            t.toFloatArray();
            fail(what + " should be closed");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("closed"));
        }
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
