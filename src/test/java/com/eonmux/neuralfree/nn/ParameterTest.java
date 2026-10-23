package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientFunction;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Parameter leaves: freezing and its survival through every replace path, node outputs handed to the constructor,
 * and substitution scopes. Runs on two host reference backends.
 */
public class ParameterTest {
    private ComputeBackend backend;
    private ComputeBackend other;

    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
        other = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        TrainGpu.release(backend);
        TrainGpu.release(other);
        backend.close();
        other.close();
    }

    @Test(expected = NullPointerException.class)
    public void testNullTensor() {
        new Parameter(null);
    }

    @Test(expected = NullPointerException.class)
    public void testNullTensorWithName() {
        new Parameter(null, "weight");
    }

    @Test
    public void testParameterNullName() {
        try {
            new Parameter(null, null);
            fail("Should throw NullPointerException");
        } catch (NullPointerException e) {
        }
    }

    @Test
    public void newParameterIsATrainableLeaf() {
        try (Parameter p = new Parameter(values(1f, 2f), "w")) {
            assertTrue(p.requiresGrad());
            assertTrue(p.tensor().requiresGrad());
            assertNull(p.tensor().gradFn());
        }
    }

    @Test
    public void parameterBuiltFromANodeOutputIsALeaf() {
        Tensor input = leaf(3f, 4f);
        Tensor output = recordedIdentity(input);
        try (Parameter p = new Parameter(output, "w")) {
            assertNotSame(output, p.tensor());
            assertTrue(p.tensor().requiresGrad());
            assertNull(p.tensor().gradFn());
            assertArrayEquals(new float[]{3f, 4f}, p.tensor().toFloatArray(), 0f);
        } finally {
            ComputationGraph.current().release(output);
            input.close();
        }
    }

    @Test
    public void freezingClosesTheGradientAndUnfreezingRestoresTheFlag() {
        try (Parameter p = new Parameter(values(1f, 2f), "w")) {
            Tensor grad = values(0.5f, 0.5f);
            p.setGrad(grad);

            p.setRequiresGrad(false);

            assertFalse(p.requiresGrad());
            assertFalse(p.tensor().requiresGrad());
            assertNull(p.tensor().gradFn());
            assertNull(p.grad());
            assertThrows(IllegalStateException.class, grad::toFloatArray);
            p.zeroGrad();
            try (Tensor rejected = values(1f, 1f)) {
                assertThrows(IllegalStateException.class, () -> p.setGrad(rejected));
            }

            p.setRequiresGrad(true);
            assertTrue(p.requiresGrad());
            assertTrue(p.tensor().requiresGrad());
            assertArrayEquals(new float[]{1f, 2f}, p.tensor().toFloatArray(), 0f);
        }
    }

    @Test
    public void freezeSurvivesMovesLoadsAndCopies() {
        try (Parameter p = new Parameter(values(1f, 2f), "w")) {
            p.setRequiresGrad(false);

            p.moveTo(other);
            assertFrozenLeaf(p);
            assertSame(other, p.backend());

            p.load(new float[]{5f, 6f});
            assertFrozenLeaf(p);
            assertArrayEquals(new float[]{5f, 6f}, p.tensor().toFloatArray(), 0f);

            try (Tensor source = Tensor.fromArray(other, new float[]{7f, 8f})) {
                p.copyFrom(source);
            }
            assertFrozenLeaf(p);
            assertArrayEquals(new float[]{7f, 8f}, p.tensor().toFloatArray(), 0f);

            try (Parameter moved = p.to(backend); Parameter copied = p.copy(); Parameter detached = p.detach()) {
                assertFrozenLeaf(moved);
                assertFrozenLeaf(copied);
                assertFrozenLeaf(detached);
            }
        }
    }

    @Test
    public void copyFromANodeOutputIsAnAssignment() {
        Tensor input = leaf(9f, 10f);
        Tensor output = recordedIdentity(input);
        try (Parameter p = new Parameter(values(1f, 2f), "w")) {
            Tensor before = p.tensor();
            p.copyFrom(output);
            assertSame(before, p.tensor());
            assertNull(p.tensor().gradFn());
            assertArrayEquals(new float[]{9f, 10f}, p.tensor().toFloatArray(), 0f);
        } finally {
            ComputationGraph.current().release(output);
            output.close();
            input.close();
        }
    }

    @Test
    public void substituteReturnsTheValueAndKeepsTheGradientSlot() {
        try (Parameter p = new Parameter(values(1f, 2f), "w"); Tensor value = values(3f, 4f)) {
            Tensor owned = p.tensor();
            Tensor grad = values(0.25f, 0.25f);
            p.setGrad(grad);
            try (Parameter.Substitution scope = p.substitute(value)) {
                assertSame(value, p.tensor());
                assertSame(grad, p.grad());
                assertSame(grad, owned.grad());
                try (Tensor inner = values(5f, 6f); Parameter.Substitution nested = p.substitute(inner)) {
                    assertSame("the inner scope shadows the outer one", inner, p.tensor());
                    assertSame(grad, p.grad());
                }
                assertSame("closing the inner scope restores the outer value", value, p.tensor());
                assertThrows(IllegalStateException.class, () -> p.moveTo(other));
            }
            assertSame(owned, p.tensor());
            assertArrayEquals(new float[]{3f, 4f}, value.toFloatArray(), 0f);
        }
    }

    @Test
    public void substitutionScopesClosedOutOfOrderRemoveOnlyTheirOwnValue() {
        try (Parameter p = new Parameter(values(1f, 2f), "w");
             Tensor outer = values(3f, 4f);
             Tensor inner = values(5f, 6f)) {
            Tensor owned = p.tensor();
            Parameter.Substitution outerScope = p.substitute(outer);
            Parameter.Substitution innerScope = p.substitute(inner);
            assertSame(inner, p.tensor());
            outerScope.close();
            assertSame("closing the outer scope first keeps the inner value", inner, p.tensor());
            innerScope.close();
            assertSame(owned, p.tensor());
            innerScope.close();
            assertSame("closing again changes nothing", owned, p.tensor());
        }
    }

    @Test
    public void substituteRestoresOnException() {
        try (Parameter p = new Parameter(values(1f, 2f), "w"); Tensor value = values(3f, 4f)) {
            Tensor owned = p.tensor();
            try (Parameter.Substitution scope = p.substitute(value)) {
                assertSame(value, p.tensor());
                throw new IllegalStateException("forward failed");
            } catch (IllegalStateException expected) {
                assertEquals("forward failed", expected.getMessage());
            }
            assertSame(owned, p.tensor());
        }
    }

    @Test
    public void substituteRejectsAnotherShape() {
        try (Parameter p = new Parameter(values(1f, 2f), "w"); Tensor wrong = values(1f, 2f, 3f)) {
            assertThrows(IllegalArgumentException.class, () -> p.substitute(wrong));
            try (Tensor onOtherBackend = Tensor.fromArray(other, new float[]{1f, 2f})) {
                assertThrows(IllegalArgumentException.class, () -> p.substitute(onOtherBackend));
            }
            try (Tensor value = values(5f, 6f); Parameter.Substitution scope = p.substitute(value)) {
                assertSame(value, p.tensor());
            }
        }
    }

    private static void assertFrozenLeaf(Parameter p) {
        assertFalse(p.requiresGrad());
        assertFalse(p.tensor().requiresGrad());
        assertNull(p.tensor().gradFn());
        assertNull(p.grad());
    }

    private Tensor values(float... data) {
        return Tensor.fromArray(backend, data);
    }

    /** A leaf that requires grad over {@code data}. */
    private Tensor leaf(float... data) {
        try (Tensor raw = values(data)) {
            return raw.withRequiresGrad(true);
        }
    }

    /** A new handle of {@code input} recorded as the output of an identity node of this thread's graph. */
    static Tensor recordedIdentity(Tensor input) {
        Tensor output;
        try (Tensor handle = input.detach()) {
            output = handle.withRequiresGrad(true);
        }
        GradientFunction identity = new GradientFunction() {
            @Override
            public List<Tensor> backward(Tensor gradOutput) {
                return Collections.singletonList(gradOutput);
            }

            @Override
            public List<Tensor> savedTensors() {
                return Collections.emptyList();
            }
        };
        output.setGradFn(identity);
        ComputationGraph.current().record(output, identity, Collections.singletonList(input));
        return output;
    }
}
