package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

public class ModuleTest {
    private ComputeBackend backend;

    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        TrainGpu.release(backend);
        backend.close();
    }

    /** Sets a scalar auxiliary loss term of {@code value} on every forward and remembers the last one. */
    static class AuxiliaryModule extends Module {
        private final ComputeBackend backend;
        private final float value;
        private Tensor lastTerm;

        AuxiliaryModule(ComputeBackend backend, float value) {
            this.backend = backend;
            this.value = value;
        }

        @Override
        public Tensor forward(Tensor input) {
            lastTerm = Tensor.scalar(backend, value);
            setAuxiliaryLoss(lastTerm);
            return input;
        }
    }

    /**
     * Sets, on every forward, a scalar term of {@code value} that is the output of a recorded identity node whose input
     * is a leaf; runs its optional child afterwards. The leaves are closed with the module.
     */
    static class RecordedAuxiliaryModule extends Module {
        private final ComputeBackend backend;
        private final float value;
        private final RecordedAuxiliaryModule child;
        private final List<Tensor> leaves = new ArrayList<>();

        RecordedAuxiliaryModule(ComputeBackend backend, float value, RecordedAuxiliaryModule child) {
            this.backend = backend;
            this.value = value;
            this.child = child == null ? null : registerModule("child", child);
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor leaf;
            try (Tensor raw = Tensor.scalar(backend, value)) {
                leaf = raw.withRequiresGrad(true);
            }
            leaves.add(leaf);
            setAuxiliaryLoss(ParameterTest.recordedIdentity(leaf));
            if (child != null) {
                child.forward(input);
            }
            return input;
        }

        @Override
        public void close() {
            try {
                super.close();
            } finally {
                for (Tensor leaf : leaves) {
                    leaf.close();
                }
                leaves.clear();
            }
        }
    }

    /** One child registered under two names (weight sharing). */
    static class SharedChildren extends Module {
        SharedChildren(Module child) {
            registerModule("first", child);
            registerModule("second", child);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    /** Runs both children; the parent itself sets a term as well. */
    static class AuxiliaryParent extends AuxiliaryModule {
        private final AuxiliaryModule first;
        private final AuxiliaryModule second;

        AuxiliaryParent(ComputeBackend backend) {
            super(backend, 1f);
            this.first = registerModule("first", new AuxiliaryModule(backend, 2f));
            this.second = registerModule("second", new AuxiliaryModule(backend, 4f));
        }

        @Override
        public Tensor forward(Tensor input) {
            super.forward(input);
            first.forward(input);
            second.forward(input);
            return input;
        }
    }
    
    static class PassthroughModule extends Module {
        @Override
        public com.eonmux.neuralfree.tensor.Tensor forward(com.eonmux.neuralfree.tensor.Tensor input) {
            return input;
        }
    }
    
    static class ParameterModule extends Module {
        private final Parameter weight;
        
        ParameterModule(String name) {
            this.weight = null;
        }
        
        @Override
        public com.eonmux.neuralfree.tensor.Tensor forward(com.eonmux.neuralfree.tensor.Tensor input) {
            return input;
        }
    }
    
    @Test
    public void testModuleRegistration() {
        PassthroughModule module = new PassthroughModule();
        assertNotNull(module);
        assertTrue(module.isTraining());
    }
    
    @Test
    public void testSequentialCreation() {
        Sequential seq = new Sequential();
        assertNotNull(seq);
        assertEquals(0, seq.size());
        assertTrue(seq.isEmpty());
    }
    
    @Test
    public void testSequentialWithModules() {
        Sequential seq = new Sequential(
            new PassthroughModule(),
            new PassthroughModule(),
            new PassthroughModule()
        );
        
        assertEquals(3, seq.size());
        assertFalse(seq.isEmpty());
        assertNotNull(seq.get(0));
        assertNotNull(seq.get(1));
        assertNotNull(seq.get(2));
    }
    
    @Test
    public void testSequentialAdd() {
        Sequential seq = new Sequential();
        seq.add(new PassthroughModule());
        seq.add(new PassthroughModule());
        
        assertEquals(2, seq.size());
    }
    
    @Test
    public void testModuleListCreation() {
        ModuleList list = new ModuleList();
        assertNotNull(list);
        assertEquals(0, list.size());
        assertTrue(list.isEmpty());
    }
    
    @Test
    public void testModuleListAdd() {
        ModuleList list = new ModuleList();
        list.add(new PassthroughModule());
        list.add(new PassthroughModule());
        
        assertEquals(2, list.size());
        assertNotNull(list.get(0));
        assertNotNull(list.get(1));
    }
    
    @Test
    public void testModuleListWithModules() {
        ModuleList list = new ModuleList(
            new PassthroughModule(),
            new PassthroughModule()
        );
        
        assertEquals(2, list.size());
    }
    
    @Test
    public void testModuleDictCreation() {
        ModuleDict dict = new ModuleDict();
        assertNotNull(dict);
        assertEquals(0, dict.size());
        assertTrue(dict.isEmpty());
    }
    
    @Test
    public void testModuleDictPut() {
        ModuleDict dict = new ModuleDict();
        dict.put("layer1", new PassthroughModule());
        dict.put("layer2", new PassthroughModule());
        
        assertEquals(2, dict.size());
        assertTrue(dict.containsKey("layer1"));
        assertTrue(dict.containsKey("layer2"));
        assertNotNull(dict.get("layer1"));
        assertNotNull(dict.get("layer2"));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testModuleDictInvalidKey() {
        ModuleDict dict = new ModuleDict();
        dict.put("invalid.key", new PassthroughModule());
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testModuleDictEmptyKey() {
        ModuleDict dict = new ModuleDict();
        dict.put("", new PassthroughModule());
    }
    
    @Test
    public void testTrainEvalMode() {
        PassthroughModule module = new PassthroughModule();
        
        assertTrue(module.isTraining());
        
        module.eval();
        assertFalse(module.isTraining());
        
        module.train();
        assertTrue(module.isTraining());
        
        module.train(false);
        assertFalse(module.isTraining());
    }
    
    @Test
    public void testTrainEvalRecursive() {
        Sequential seq = new Sequential(
            new PassthroughModule(),
            new PassthroughModule()
        );
        
        assertTrue(seq.isTraining());
        
        seq.eval();
        assertFalse(seq.isTraining());
        for (Module child : seq.children()) {
            assertFalse(child.isTraining());
        }
        
        seq.train();
        assertTrue(seq.isTraining());
        for (Module child : seq.children()) {
            assertTrue(child.isTraining());
        }
    }
    
    @Test
    public void testModuleChildren() {
        Sequential seq = new Sequential(
            new PassthroughModule(),
            new PassthroughModule()
        );
        
        List<Module> children = seq.children();
        assertEquals(2, children.size());
    }
    
    @Test
    public void testModuleNamedChildren() {
        ModuleDict dict = new ModuleDict();
        dict.put("a", new PassthroughModule());
        dict.put("b", new PassthroughModule());
        
        Map<String, Module> namedChildren = dict.namedChildren();
        assertEquals(2, namedChildren.size());
        assertTrue(namedChildren.containsKey("a"));
        assertTrue(namedChildren.containsKey("b"));
    }
    
    @Test
    public void testModules() {
        Sequential seq = new Sequential(
            new PassthroughModule(),
            new PassthroughModule()
        );
        
        List<Module> modules = seq.modules();
        assertEquals(3, modules.size());
    }
    
    @Test
    public void testNestedParameters() {
        Sequential seq = new Sequential();
        seq.add(new PassthroughModule());
        seq.add(new PassthroughModule());
        
        List<Parameter> params = seq.parameters();
        assertEquals(0, params.size());
    }
    
    @Test
    public void testModuleName() {
        PassthroughModule module = new PassthroughModule();
        assertEquals("PassthroughModule", module.moduleName());
    }
    
    @Test
    public void testSequentialModuleName() {
        Sequential seq = new Sequential();
        assertEquals("Sequential", seq.moduleName());
    }
    
    @Test
    public void testModuleListModuleName() {
        ModuleList list = new ModuleList();
        assertEquals("ModuleList", list.moduleName());
    }
    
    @Test
    public void testModuleDictModuleName() {
        ModuleDict dict = new ModuleDict();
        assertEquals("ModuleDict", dict.moduleName());
    }
    
    @Test
    public void testParameterCount() {
        PassthroughModule module = new PassthroughModule();
        assertEquals(0, module.parameterCount());
    }
    
    @Test
    public void testToString() {
        PassthroughModule module = new PassthroughModule();
        String str = module.toString();
        assertTrue(str.contains("PassthroughModule"));
        assertTrue(str.contains("training=true"));
    }
    
    @Test
    public void testSequentialToString() {
        Sequential seq = new Sequential(new PassthroughModule());
        String str = seq.toString();
        assertTrue(str.contains("Sequential"));
    }
    
    @Test
    public void testModuleListImplementsList() {
        ModuleList list = new ModuleList();
        assertTrue(list instanceof java.util.List);
    }
    
    @Test
    public void testModuleDictImplementsMap() {
        ModuleDict dict = new ModuleDict();
        assertTrue(dict instanceof java.util.Map);
    }
    
    @Test(expected = IllegalStateException.class)
    public void testClosedModuleThrows() {
        PassthroughModule module = new PassthroughModule();
        module.close();
        module.parameters();
    }
    
    @Test
    public void testModuleCloseIsIdempotent() {
        PassthroughModule module = new PassthroughModule();
        module.close();
        module.close();
    }
    
    @Test
    public void testContainsKey() {
        ModuleDict dict = new ModuleDict();
        dict.put("test", new PassthroughModule());
        assertTrue(dict.containsKey("test"));
        assertFalse(dict.containsKey("other"));
    }
    
    @Test
    public void testKeySet() {
        ModuleDict dict = new ModuleDict();
        dict.put("a", new PassthroughModule());
        dict.put("b", new PassthroughModule());
        
        java.util.Set<String> keys = dict.keySet();
        assertEquals(2, keys.size());
        assertTrue(keys.contains("a"));
        assertTrue(keys.contains("b"));
    }
    
    @Test
    public void testValues() {
        ModuleDict dict = new ModuleDict();
        PassthroughModule m1 = new PassthroughModule();
        PassthroughModule m2 = new PassthroughModule();
        dict.put("a", m1);
        dict.put("b", m2);
        
        java.util.Collection<Module> values = dict.values();
        assertEquals(2, values.size());
        assertTrue(values.contains(m1));
        assertTrue(values.contains(m2));
    }
    
    @Test
    public void testIterator() {
        ModuleList list = new ModuleList();
        list.add(new PassthroughModule());
        list.add(new PassthroughModule());
        
        int count = 0;
        for (Module m : list) {
            assertNotNull(m);
            count++;
        }
        assertEquals(2, count);
    }
    
    @Test(expected = UnsupportedOperationException.class)
    public void testModuleDictRemove() {
        ModuleDict dict = new ModuleDict();
        dict.put("test", new PassthroughModule());
        dict.remove("test");
    }
    
    @Test(expected = UnsupportedOperationException.class)
    public void testModuleListRemove() {
        ModuleList list = new ModuleList();
        list.add(new PassthroughModule());
        list.remove(0);
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSequentialGetOutOfBounds() {
        Sequential seq = new Sequential();
        seq.get(0);
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testModuleListGetOutOfBounds() {
        ModuleList list = new ModuleList();
        list.get(0);
    }
    
    @Test(expected = NullPointerException.class)
    public void testNullParameterRegistration() {
        PassthroughModule module = new PassthroughModule();
        module.registerParameter("weight", null);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testDuplicateModuleName() {
        PassthroughModule parent = new PassthroughModule();
        parent.registerModule("child", new PassthroughModule());
        parent.registerModule("child", new PassthroughModule());
    }
    
    @Test
    public void testSequentialVarargConstructor() {
        Sequential seq = new Sequential(
            new PassthroughModule(),
            new PassthroughModule(),
            new PassthroughModule()
        );
        assertEquals(3, seq.size());
    }
    
    @Test
    public void testModuleListVarargConstructor() {
        ModuleList list = new ModuleList(
            new PassthroughModule(),
            new PassthroughModule()
        );
        assertEquals(2, list.size());
    }
    
    @Test
    public void testModuleDictMapConstructor() {
        Map<String, Module> modules = new java.util.HashMap<>();
        modules.put("a", new PassthroughModule());
        modules.put("b", new PassthroughModule());
        
        ModuleDict dict = new ModuleDict(modules);
        assertEquals(2, dict.size());
    }
    
    @Test
    public void testNamedModules() {
        Sequential seq = new Sequential();
        seq.add(new PassthroughModule());
        seq.add(new PassthroughModule());
        
        Map<String, Module> named = seq.namedModules();
        assertTrue(named.containsKey("0"));
        assertTrue(named.containsKey("1"));
    }
    
    @Test
    public void testLayerIteration() {
        Sequential seq = new Sequential(
            new PassthroughModule(),
            new PassthroughModule(),
            new PassthroughModule()
        );
        
        List<Module> layers = seq.layers();
        assertEquals(3, layers.size());
    }
    
    @Test(expected = NullPointerException.class)
    public void testSequentialNullModule() {
        Sequential seq = new Sequential();
        seq.add(null);
    }
    
    @Test(expected = NullPointerException.class)
    public void testModuleListNullModule() {
        ModuleList list = new ModuleList();
        list.add(null);
    }
    
    @Test(expected = NullPointerException.class)
    public void testModuleDictNullKey() {
        ModuleDict dict = new ModuleDict();
        dict.put(null, new PassthroughModule());
    }
    
    @Test(expected = NullPointerException.class)
    public void testModuleDictNullModule() {
        ModuleDict dict = new ModuleDict();
        dict.put("test", null);
    }

    @Test
    public void frozenModuleForwardDoesNotRequireGrad() {
        try (Linear linear = linear(1L); Tensor input = Tensor.fromArray(backend, new float[][]{{1f, 2f, 3f}})) {
            assertTrue(ComputationGraph.isGradEnabled());
            linear.setRequiresGrad(false);
            for (Parameter p : linear.parameters()) {
                assertFalse(p.requiresGrad());
                assertFalse(p.tensor().requiresGrad());
            }
            Tensor frozenOut = linear.forward(input);
            try {
                assertFalse(frozenOut.requiresGrad());
                assertNull(frozenOut.gradFn());
            } finally {
                release(frozenOut);
            }

            linear.setRequiresGrad(true);
            Tensor trainableOut = linear.forward(input);
            try {
                assertTrue(trainableOut.requiresGrad());
            } finally {
                release(trainableOut);
            }
        }
    }

    @Test
    public void setRequiresGradCoversDescendants() {
        Sequential seq = new Sequential(linear(2L), linear(3L));
        try {
            assertSame(seq, seq.setRequiresGrad(false));
            assertEquals(4, seq.parameters().size());
            for (Parameter p : seq.parameters()) {
                assertFalse(p.requiresGrad());
                assertNull(p.tensor().gradFn());
            }
            seq.setRequiresGrad(true);
            for (Parameter p : seq.parameters()) {
                assertTrue(p.requiresGrad());
            }
        } finally {
            seq.close();
        }
    }

    @Test
    public void substituteReplacesParametersForTheScopeOnly() {
        try (Linear linear = linear(4L); Tensor input = Tensor.fromArray(backend, new float[][]{{1f, 2f, 3f}});
             Tensor zeroWeight = Tensor.fromArray(backend, new float[][]{{0f, 0f, 0f}, {0f, 0f, 0f}})) {
            Parameter weight = linear.weight();
            Tensor owned = weight.tensor();
            Map<Parameter, Tensor> values = new IdentityHashMap<>();
            values.put(weight, zeroWeight);
            try (Parameter.Substitution scope = linear.substitute(values)) {
                assertSame(zeroWeight, weight.tensor());
                Tensor out = linear.forward(input);
                try {
                    assertArrayEquals(linear.bias().tensor().toFloatArray(), out.toFloatArray(), 0f);
                } finally {
                    release(out);
                }
            }
            assertSame(owned, weight.tensor());
        }
    }

    @Test
    public void substituteRestoresOnException() {
        try (Linear linear = linear(5L);
             Tensor zeroWeight = Tensor.fromArray(backend, new float[][]{{0f, 0f, 0f}, {0f, 0f, 0f}});
             Tensor wrongBias = Tensor.fromArray(backend, new float[]{0f, 0f, 0f})) {
            Parameter weight = linear.weight();
            Tensor owned = weight.tensor();
            Map<Parameter, Tensor> values = new IdentityHashMap<>();
            values.put(weight, zeroWeight);
            try (Parameter.Substitution scope = linear.substitute(values)) {
                throw new IllegalStateException("forward failed");
            } catch (IllegalStateException expected) {
                assertEquals("forward failed", expected.getMessage());
            }
            assertSame(owned, weight.tensor());

            values.put(linear.bias(), wrongBias);
            assertThrows(IllegalArgumentException.class, () -> linear.substitute(values));
            assertSame(owned, weight.tensor());
        }
    }

    @Test
    public void substituteRejectsForeignParameters() {
        try (Linear linear = linear(6L); Linear foreign = linear(7L)) {
            Map<Parameter, Tensor> values = new IdentityHashMap<>();
            values.put(foreign.weight(), foreign.weight().tensor());
            assertThrows(IllegalArgumentException.class, () -> linear.substitute(values));
        }
    }

    @Test
    public void takeAuxiliaryLossSumsEveryForwardOfTheTreeAndClears() {
        try (AuxiliaryParent parent = new AuxiliaryParent(backend);
             Tensor input = Tensor.scalar(backend, 0f)) {
            assertNull(parent.takeAuxiliaryLoss());

            parent.forward(input);
            Tensor folded = parent.first.lastTerm;
            parent.forward(input);
            assertThrows("a constant term is folded into the pending constant and closed", IllegalStateException.class,
                folded::toFloatArray);

            Tensor total = parent.takeAuxiliaryLoss();
            try {
                assertNotNull(total);
                assertEquals(1, total.numElements());
                assertEquals("both forwards' terms are summed", 14f, total.toFloatArray()[0], 0f);
            } finally {
                release(total);
            }
            assertNull(parent.takeAuxiliaryLoss());
            assertNull(parent.first.takeAuxiliaryLoss());
        }
    }

    @Test
    public void recordedTermsOfEveryForwardAccumulateAndAllReceiveGradients() {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        try (RecordedAuxiliaryModule parent = new RecordedAuxiliaryModule(backend, 2f,
                new RecordedAuxiliaryModule(backend, 3f, null));
             Tensor input = Tensor.scalar(backend, 0f)) {
            parent.forward(input);
            parent.forward(input);
            assertEquals("two forwards of two modules keep four terms", before + 4, graph.size());

            try (Tensor total = parent.takeAuxiliaryLoss()) {
                assertTrue("the sum is recorded", total.requiresGrad());
                assertNotNull(total.gradFn());
                assertEquals(10f, total.toFloatArray()[0], 0f);
                total.backward();
            }
            assertEquals("one backward releases every term and the sum", before, graph.size());
            List<Tensor> leaves = new ArrayList<>(parent.leaves);
            leaves.addAll(parent.child.leaves);
            assertEquals(4, leaves.size());
            for (Tensor leaf : leaves) {
                assertNotNull("every forward's term delivers a gradient", leaf.grad());
                assertEquals(1f, leaf.grad().toFloatArray()[0], 0f);
            }
            assertNull(parent.takeAuxiliaryLoss());
        } finally {
            graph.zeroGrad();
        }
    }

    @Test
    public void aTermWhoseNodeAStepScopeDiscardedIsDropped() {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        try (RecordedAuxiliaryModule module = new RecordedAuxiliaryModule(backend, 2f, null);
             Tensor input = Tensor.scalar(backend, 0f)) {
            module.forward(input);
            try (ComputationGraph.StepScope step = graph.step()) {
                module.forward(input);
                assertEquals(before + 2, graph.size());
            }
            assertEquals("the scope discarded its term's node", before + 1, graph.size());

            Tensor total = module.takeAuxiliaryLoss();
            try {
                assertNotNull(total);
                assertNotNull("the surviving term is handed over as is", graph.getNode(total));
                assertEquals("the discarded term is not summed", 2f, total.toFloatArray()[0], 0f);
            } finally {
                release(total);
            }
            assertEquals(before, graph.size());
            assertNull(module.takeAuxiliaryLoss());
        }
    }

    @Test
    public void clearAuxiliaryLossReleasesEveryPendingTerm() {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        try (RecordedAuxiliaryModule parent = new RecordedAuxiliaryModule(backend, 2f,
                new RecordedAuxiliaryModule(backend, 3f, null));
             Tensor input = Tensor.scalar(backend, 0f)) {
            parent.forward(input);
            assertEquals(before + 2, graph.size());
            parent.clearAuxiliaryLoss();
            assertEquals(before, graph.size());
            assertNull(parent.takeAuxiliaryLoss());
        }
    }

    @Test
    public void setAuxiliaryLossRejectsNullAndATermAlreadyPending() {
        try (AuxiliaryModule module = new AuxiliaryModule(backend, 0f);
             Tensor term = Tensor.scalar(backend, 5f)) {
            assertThrows(NullPointerException.class, () -> module.setAuxiliaryLoss(null));
            module.setAuxiliaryLoss(term);
            assertThrows(IllegalArgumentException.class, () -> module.setAuxiliaryLoss(term));
            Tensor total = module.takeAuxiliaryLoss();
            assertSame(term, total);
            assertEquals(5f, total.toFloatArray()[0], 0f);
        }
    }

    @Test
    public void takeAuxiliaryLossRefusesATermRecordedOnAnotherThread() throws InterruptedException {
        try (RecordedAuxiliaryModule module = new RecordedAuxiliaryModule(backend, 2f, null);
             Tensor input = Tensor.scalar(backend, 0f)) {
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread worker = new Thread(() -> {
                try {
                    module.forward(input);
                } catch (Throwable t) {
                    failure.set(t);
                }
            });
            worker.start();
            worker.join();
            assertNull(failure.get());
            assertThrows(IllegalStateException.class, module::takeAuxiliaryLoss);
            module.clearAuxiliaryLoss();
            assertNull(module.takeAuxiliaryLoss());
        }
    }

    @Test
    public void sharedModulesAndParametersAreListedOnce() {
        try (SharedChildren shared = new SharedChildren(linear(9L))) {
            assertEquals(2, shared.parameters().size());
            assertEquals(2, shared.modules().size());
            assertEquals(4, shared.namedParameters().size());
            assertTrue(shared.namedParameters().containsKey("first.weight"));
            assertTrue(shared.namedParameters().containsKey("second.weight"));
            assertEquals(3 * 2 + 2, shared.parameterCount());
        }
    }

    @Test
    public void takeAuxiliaryLossOfOneTermHandsItOver() {
        try (AuxiliaryModule module = new AuxiliaryModule(backend, 3f);
             Tensor input = Tensor.scalar(backend, 0f)) {
            module.forward(input);
            Tensor total = module.takeAuxiliaryLoss();
            try {
                assertSame(module.lastTerm, total);
                assertEquals(3f, total.toFloatArray()[0], 0f);
            } finally {
                release(total);
            }
        }
    }

    @Test
    public void closeReleasesThePendingAuxiliaryTerm() {
        AuxiliaryModule module = new AuxiliaryModule(backend, 3f);
        try (Tensor input = Tensor.scalar(backend, 0f)) {
            module.forward(input);
        }
        Tensor pending = module.lastTerm;
        module.close();
        assertThrows(IllegalStateException.class, pending::toFloatArray);
    }

    @Test
    public void auxiliaryLossMustBeAScalar() {
        try (AuxiliaryModule module = new AuxiliaryModule(backend, 0f);
             Tensor vector = Tensor.fromArray(backend, new float[]{1f, 2f})) {
            assertThrows(IllegalArgumentException.class, () -> module.setAuxiliaryLoss(vector));
            assertNull(module.takeAuxiliaryLoss());
        }
    }

    @Test
    public void takeAuxiliaryLossUnderNoGradReleasesRecordedTerms() {
        int before = ComputationGraph.current().size();
        try (RecordedAuxiliaryModule parent = new RecordedAuxiliaryModule(backend, 2f,
                new RecordedAuxiliaryModule(backend, 3f, null));
             Tensor input = Tensor.scalar(backend, 0f)) {
            parent.forward(input);
            assertEquals(before + 2, ComputationGraph.current().size());

            Tensor total;
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                total = parent.takeAuxiliaryLoss();
            }
            try {
                assertNull(total.gradFn());
                assertEquals(5f, total.toFloatArray()[0], 0f);
                assertEquals(before, ComputationGraph.current().size());
            } finally {
                release(total);
            }
            assertNull(parent.takeAuxiliaryLoss());
        }
        assertEquals(before, ComputationGraph.current().size());
    }

    @Test
    public void stateDictReadsTheOwnedValuesWhileSubstituted() {
        try (Linear linear = linear(8L);
             Tensor zeroWeight = Tensor.fromArray(backend, new float[][]{{0f, 0f, 0f}, {0f, 0f, 0f}})) {
            Parameter weight = linear.weight();
            float[] owned = weight.tensor().toFloatArray();
            Map<Parameter, Tensor> values = new IdentityHashMap<>();
            values.put(weight, zeroWeight);
            try (Parameter.Substitution scope = linear.substitute(values)) {
                assertArrayEquals(owned, (float[]) linear.stateDict().get("weight"), 0f);
            }
        }
    }

    private Linear linear(long seed) {
        Linear linear = new Linear(3, 2);
        linear.initialize(backend, new Random(seed));
        return linear;
    }

    /** Releases the graph {@code t} was computed from, if any, and closes it. */
    private static void release(Tensor t) {
        try {
            ComputationGraph.current().release(t);
        } finally {
            t.close();
        }
    }
}
