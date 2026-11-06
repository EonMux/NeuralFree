package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The thread's {@link ComputationGraph} remembers the leaves it gave gradients to without keeping them reachable:
 * a leaf its owner dropped is collected even though the graph outlives it (the graph lives as long as its thread),
 * while a live leaf still has its gradient closed by {@link ComputationGraph#zeroGrad()}. Also covers the
 * {@link WeakIdentitySet} behind it.
 */
public class GradientLeafRetentionTest {

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
        backend.close();
    }

    @Test
    public void theGraphDoesNotKeepADroppedLeafReachable() throws InterruptedException {
        WeakReference<Tensor> dropped = trainAndDropLeaf();
        awaitCleared(dropped);
        assertNull("the thread's graph kept a leaf its owner dropped reachable", dropped.get());
    }

    @Test
    public void zeroGradStillClosesTheGradientOfALiveLeaf() {
        try (Tensor x = leaf(1f, 2f); Tensor seed = Tensor.fromArray(backend, new float[]{0.5f, -2f})) {
            graph.backward(x, seed);
            Tensor gradient = x.grad();
            assertArrayEquals(new float[]{0.5f, -2f}, gradient.toFloatArray(), 0f);
            graph.zeroGrad();
            assertNull("the graph clears the slot of a leaf it gave a gradient to", x.grad());
            AutogradRecorderTest.assertClosed(gradient, "the cleared gradient");
        }
    }

    @Test
    public void theSetComparesByIdentity() {
        WeakIdentitySet<String> set = new WeakIdentitySet<>();
        String first = new String("same");
        String second = new String("same");
        assertTrue(set.add(first));
        assertTrue("an equal but distinct element is another member", set.add(second));
        assertFalse("the same element is added once", set.add(first));
        assertEquals(2, set.size());
        assertTrue(set.contains(first));
        assertFalse(set.contains(new String("same")));
        assertFalse(set.contains(null));
        assertTrue(set.remove(first));
        assertFalse(set.contains(first));
        assertTrue(set.contains(second));
        assertFalse(set.remove(null));
    }

    @Test
    public void theIteratorRemovesFromTheSet() {
        WeakIdentitySet<Object> set = new WeakIdentitySet<>();
        Object kept = new Object();
        Object removed = new Object();
        set.add(kept);
        set.add(removed);
        Iterator<Object> it = set.iterator();
        List<Object> seen = new ArrayList<>();
        while (it.hasNext()) {
            Object next = it.next();
            seen.add(next);
            if (next == removed) {
                it.remove();
            }
        }
        assertEquals(2, seen.size());
        assertEquals(1, set.size());
        assertSame(kept, set.iterator().next());
        assertTrue(set.removeIf(o -> o == kept));
        assertTrue(set.isEmpty());
        try {
            set.iterator().remove();
            fail("remove before next must refuse");
        } catch (IllegalStateException expected) {
            // no element was returned yet
        }
    }

    @Test(expected = NullPointerException.class)
    public void theSetRefusesNull() {
        new WeakIdentitySet<Object>().add(null);
    }

    @Test
    public void aCollectedElementLeavesTheSet() throws InterruptedException {
        WeakIdentitySet<Object> set = new WeakIdentitySet<>();
        Object kept = new Object();
        set.add(kept);
        WeakReference<Object> dropped = addDropped(set);
        awaitCleared(dropped);
        assertNull(dropped.get());
        assertEquals("only the live element stays", 1, set.size());
        assertSame(kept, set.iterator().next());
        set.clear();
        assertTrue(set.isEmpty());
    }

    /** Runs a leaf backward, then releases the leaf as its owner would: clears and closes the gradient, closes it. */
    private WeakReference<Tensor> trainAndDropLeaf() {
        Tensor x = leaf(1f, 2f);
        try (Tensor seed = Tensor.fromArray(backend, new float[]{1f, 1f})) {
            graph.backward(x, seed);
        }
        Tensor gradient = x.grad();
        x.setGrad(null);
        gradient.close();
        x.close();
        return new WeakReference<>(x);
    }

    private static WeakReference<Object> addDropped(WeakIdentitySet<Object> set) {
        Object element = new Object();
        set.add(element);
        return new WeakReference<>(element);
    }

    /** Requests collections until {@code ref} is cleared, bounded. */
    private static void awaitCleared(WeakReference<?> ref) throws InterruptedException {
        for (int i = 0; i < 50 && ref.get() != null; i++) {
            System.gc();
            Thread.sleep(20);
        }
    }

    private Tensor leaf(float... values) {
        try (Tensor t = Tensor.fromArray(backend, values)) {
            return t.withRequiresGrad(true);
        }
    }
}
