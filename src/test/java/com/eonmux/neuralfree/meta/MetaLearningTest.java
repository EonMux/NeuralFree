package com.eonmux.neuralfree.meta;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.nn.loss.MSELoss;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

public class MetaLearningTest {

    @Test
    public void testMAMLConstruction() {
        MockModule model = new MockModule();
        MSELoss loss = new MSELoss();
        
        MAML maml = new MAML(model, 0.01f, 0.001f, 5, loss);
        
        assertEquals(0.01f, maml.innerLearningRate(), 0.0001f);
        assertEquals(0.001f, maml.outerLearningRate(), 0.0001f);
        assertEquals(5, maml.innerSteps());
        assertTrue(maml.isFirstOrder());
        assertSame(model, maml.model());
    }

    @Test
    public void testMAMLFirstOrderVsSecondOrder() {
        MockModule model = new MockModule();
        MSELoss loss = new MSELoss();
        
        MAML mamlFirst = new MAML(model, 0.01f, 0.001f, 5, loss, true);
        MAML mamlSecond = new MAML(model, 0.01f, 0.001f, 5, loss, false);
        
        assertTrue(mamlFirst.isFirstOrder());
        assertFalse(mamlSecond.isFirstOrder());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMAMLInvalidInnerLearningRate() {
        new MAML(new MockModule(), 0.0f, 0.001f, 5, new MSELoss());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMAMLInvalidOuterLearningRate() {
        new MAML(new MockModule(), 0.01f, 0.0f, 5, new MSELoss());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMAMLInvalidInnerSteps() {
        new MAML(new MockModule(), 0.01f, 0.001f, 0, new MSELoss());
    }

    @Test(expected = NullPointerException.class)
    public void testMAMLNullModel() {
        new MAML(null, 0.01f, 0.001f, 5, new MSELoss());
    }

    @Test(expected = NullPointerException.class)
    public void testMAMLNullLoss() {
        new MAML(new MockModule(), 0.01f, 0.001f, 5, null);
    }

    @Test
    public void testMAMLResult() {
        List<Float> metaLosses = Arrays.asList(1.0f, 0.8f, 0.6f);
        List<Float> valLosses = Arrays.asList(1.1f, 0.9f, 0.7f);

        MAML.MAMLResult result = new MAML.MAMLResult(
            new MockModule(), metaLosses, valLosses);
        
        assertEquals(3, result.metaLosses().size());
        assertEquals(3, result.validationLosses().size());
        assertEquals(0.6f, result.finalMetaLoss(), 0.001f);
        assertEquals(0.7f, result.finalValidationLoss(), 0.001f);
        assertNotNull(result.model());
    }

    @Test
    public void testMAMLResultEmpty() {
        MAML.MAMLResult result = new MAML.MAMLResult(
            new MockModule(), new ArrayList<>(), new ArrayList<>());
        
        assertTrue(Float.isNaN(result.finalMetaLoss()));
        assertTrue(Float.isNaN(result.finalValidationLoss()));
    }

    @Test
    public void testKnowledgeDistillerConstruction() {
        MockModule teacher = new MockModule();
        MockModule student = new MockModule();
        MSELoss loss = new MSELoss();
        
        KnowledgeDistiller distiller = new KnowledgeDistiller(teacher, student, 4.0f, 0.5f, loss);
        
        assertEquals(4.0f, distiller.temperature(), 0.001f);
        assertEquals(0.5f, distiller.alpha(), 0.001f);
        assertSame(teacher, distiller.teacher());
        assertSame(student, distiller.student());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testKnowledgeDistillerInvalidTemperature() {
        new KnowledgeDistiller(
            new MockModule(), 
            new MockModule(), 
            0.0f, 0.5f, new MSELoss());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testKnowledgeDistillerInvalidAlphaHigh() {
        new KnowledgeDistiller(
            new MockModule(), 
            new MockModule(), 
            4.0f, 1.5f, new MSELoss());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testKnowledgeDistillerInvalidAlphaLow() {
        new KnowledgeDistiller(
            new MockModule(), 
            new MockModule(), 
            4.0f, -0.5f, new MSELoss());
    }

    @Test
    public void testKnowledgeDistillerResult() {
        List<Float> totalLosses = Arrays.asList(1.0f, 0.8f, 0.6f);
        List<Float> distillLosses = Arrays.asList(0.5f, 0.4f, 0.3f);
        List<Float> hardLosses = Arrays.asList(0.5f, 0.4f, 0.3f);

        KnowledgeDistiller.DistillationResult result = new KnowledgeDistiller.DistillationResult(
            new MockModule(), totalLosses, distillLosses, hardLosses);
        
        assertEquals(3, result.totalLosses().size());
        assertEquals(0.6f, result.finalTotalLoss(), 0.001f);
        assertEquals(0.3f, result.finalDistillationLoss(), 0.001f);
        assertEquals(0.3f, result.finalHardTargetLoss(), 0.001f);
        assertNotNull(result.student());
    }

    @Test
    public void testNASSearchSpaceConstruction() {
        List<NASSearchSpace.Operation> ops = NASSearchSpace.standardOperations();
        NASSearchSpace space = new NASSearchSpace(4, ops);
        
        assertEquals(4, space.numNodes());
        assertEquals(6, space.numEdges());
        assertEquals(8, space.availableOperations().size());
        assertNotNull(space.name());
    }

    @Test
    public void testNASSearchSpaceEdges() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        
        assertEquals(3, space.numEdges());
        
        List<NASSearchSpace.Edge> edges = space.edges();
        assertEquals(0, edges.get(0).sourceNode());
        assertEquals(1, edges.get(0).targetNode());
        assertEquals(0, edges.get(1).sourceNode());
        assertEquals(2, edges.get(1).targetNode());
        assertEquals(1, edges.get(2).sourceNode());
        assertEquals(2, edges.get(2).targetNode());
    }

    @Test
    public void testNASSearchSpaceArchitecture() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        
        List<Integer> arch = Arrays.asList(1, 2, 3);
        space.setArchitecture(arch);
        
        List<Integer> retrieved = space.getArchitecture();
        assertEquals(arch, retrieved);
    }

    @Test
    public void testNASSearchSpaceRandomArchitecture() {
        NASSearchSpace space = new NASSearchSpace(4, NASSearchSpace.standardOperations());
        
        space.setRandomArchitecture(42);
        List<Integer> arch1 = space.getArchitecture();
        
        space.setRandomArchitecture(42);
        List<Integer> arch2 = space.getArchitecture();
        
        assertEquals(arch1, arch2);
        
        space.setRandomArchitecture(123);
        List<Integer> arch3 = space.getArchitecture();
        
        assertNotEquals(arch1, arch3);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNASSearchSpaceInvalidNumNodes() {
        new NASSearchSpace(1, NASSearchSpace.standardOperations());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNASSearchSpaceEmptyOperations() {
        new NASSearchSpace(4, new ArrayList<>());
    }

    @Test
    public void testNASSearchSpaceCopy() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        space.setArchitecture(Arrays.asList(1, 2, 3));
        
        NASSearchSpace copy = space.copy();
        
        assertEquals(space.getArchitecture(), copy.getArchitecture());
        assertNotSame(space, copy);
    }

    @Test
    public void testNASOperation() {
        NASSearchSpace.Operation op = new NASSearchSpace.Operation(
            "conv_3x3", NASSearchSpace.OperationType.CONV_3X3);
        
        assertEquals("conv_3x3", op.name());
        assertEquals(NASSearchSpace.OperationType.CONV_3X3, op.type());
        assertNotNull(op.params());
    }

    @Test
    public void testNASOperationEquality() {
        NASSearchSpace.Operation op1 = new NASSearchSpace.Operation("conv", NASSearchSpace.OperationType.CONV_3X3);
        NASSearchSpace.Operation op2 = new NASSearchSpace.Operation("conv", NASSearchSpace.OperationType.CONV_5X5);
        NASSearchSpace.Operation op3 = new NASSearchSpace.Operation("pool", NASSearchSpace.OperationType.AVG_POOL_3X3);
        
        assertEquals(op1, op2);
        assertNotEquals(op1, op3);
        assertEquals(op1.hashCode(), op2.hashCode());
    }

    @Test
    public void testNASControllerConstruction() {
        NASSearchSpace space = new NASSearchSpace(4, NASSearchSpace.standardOperations());
        
        NASController controller = new NASController(
            space, NASController.SearchStrategy.RANDOM);
        
        assertSame(space, controller.searchSpace());
        assertEquals(NASController.SearchStrategy.RANDOM, controller.strategy());
    }

    @Test
    public void testNASControllerRandomSearch() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.RANDOM, 42);
        
        NASController.Evaluator evaluator = arch -> (float) Math.random();
        
        NASController.NASResult result = controller.search(10, evaluator);
        
        assertEquals(10, result.totalEvaluations());
        assertNotNull(result.bestArchitecture());
        assertFalse(Float.isNaN(result.bestFitness()));
    }

    @Test
    public void testNASControllerEvolutionarySearch() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.EVOLUTIONARY, 42);
        
        NASController.Evaluator evaluator = arch -> (float) Math.random();
        
        NASController.NASResult result = controller.search(60, evaluator);
        
        assertEquals(60, result.totalEvaluations());
        assertNotNull(result.bestArchitecture());
    }

    @Test
    public void testNASControllerDARTSSearch() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.DARTS, 42);
        
        NASController.Evaluator evaluator = arch -> (float) Math.random();
        
        NASController.NASResult result = controller.search(10, evaluator);
        
        assertEquals(10, result.totalEvaluations());
        assertNotNull(result.bestArchitecture());
    }

    @Test
    public void testNASControllerSetMutationRate() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.EVOLUTIONARY);
        
        controller.setMutationRate(0.2f);
        controller.setMutationRate(0.0f);
        controller.setMutationRate(1.0f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNASControllerInvalidMutationRate() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.EVOLUTIONARY);
        
        controller.setMutationRate(1.5f);
    }

    @Test
    public void testNASControllerSetLearningRate() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.DARTS);
        
        controller.setLearningRate(0.1f);
        controller.setLearningRate(0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNASControllerInvalidLearningRate() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.DARTS);
        
        controller.setLearningRate(0.0f);
    }

    @Test
    public void testFederatedMetaLearnerConstruction() {
        MockModule model = new MockModule();
        MSELoss loss = new MSELoss();
        
        FederatedMetaLearner learner = new FederatedMetaLearner(
            model, loss, 5, 10, 3, 0.01f, 0.001f);
        
        assertEquals(5, learner.numClients());
        assertEquals(10, learner.numRounds());
        assertEquals(0.1f, learner.clientFraction(), 0.001f);
        assertSame(model, learner.globalModel());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFederatedMetaLearnerInvalidNumClients() {
        new FederatedMetaLearner(
            new MockModule(), 
            new MSELoss(), 
            0, 10, 3, 0.01f, 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFederatedMetaLearnerInvalidNumRounds() {
        new FederatedMetaLearner(
            new MockModule(), 
            new MSELoss(), 
            5, 0, 3, 0.01f, 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFederatedMetaLearnerInvalidInnerSteps() {
        new FederatedMetaLearner(
            new MockModule(), 
            new MSELoss(), 
            5, 10, 0, 0.01f, 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFederatedMetaLearnerInvalidInnerLR() {
        new FederatedMetaLearner(
            new MockModule(), 
            new MSELoss(), 
            5, 10, 3, 0.0f, 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFederatedMetaLearnerInvalidMetaLR() {
        new FederatedMetaLearner(
            new MockModule(), 
            new MSELoss(), 
            5, 10, 3, 0.01f, 0.0f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFederatedMetaLearnerInvalidClientFraction() {
        new FederatedMetaLearner(
            new MockModule(), 
            new MSELoss(), 
            5, 10, 3, 0.01f, 0.001f, 0.0f);
    }

    @Test
    public void testFederatedMetaResult() {
        List<Float> losses = Arrays.asList(1.0f, 0.8f, 0.6f);
        List<Float> accuracies = Arrays.asList(0.5f, 0.7f, 0.9f);

        FederatedMetaLearner.FederatedMetaResult result = 
            new FederatedMetaLearner.FederatedMetaResult(
                new MockModule(), losses, accuracies);
        
        assertEquals(3, result.numRounds());
        assertEquals(0.6f, result.finalLoss(), 0.001f);
        assertEquals(0.9f, result.finalAccuracy(), 0.001f);
        assertNotNull(result.model());
    }

    @Test
    public void testFederatedMetaResultEmpty() {
        FederatedMetaLearner.FederatedMetaResult result = 
            new FederatedMetaLearner.FederatedMetaResult(
                new MockModule(), new ArrayList<>(), new ArrayList<>());
        
        assertTrue(Float.isNaN(result.finalLoss()));
        assertTrue(Float.isNaN(result.finalAccuracy()));
    }

    @Test
    public void testMAMLClose() {
        MAML maml = new MAML(
            new MockModule(), 0.01f, 0.001f, 5, new MSELoss());
        
        maml.close();
        
        try {
            maml.model();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
    }

    @Test
    public void testKnowledgeDistillerClose() {
        KnowledgeDistiller distiller = new KnowledgeDistiller(
            new MockModule(),
            new MockModule(),
            4.0f, 0.5f, new MSELoss());
        
        distiller.close();
        
        try {
            distiller.teacher();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
    }

    @Test
    public void testNASControllerClose() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASController controller = new NASController(
            space, NASController.SearchStrategy.RANDOM);
        
        controller.close();
    }

    @Test
    public void testFederatedMetaLearnerClose() {
        FederatedMetaLearner learner = new FederatedMetaLearner(
            new MockModule(),
            new MSELoss(),
            5, 10, 3, 0.01f, 0.001f);
        
        learner.close();
        
        try {
            learner.globalModel();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
    }

    @Test
    public void testStandardOperations() {
        List<NASSearchSpace.Operation> ops = NASSearchSpace.standardOperations();
        assertEquals(8, ops.size());
    }

    @Test
    public void testDartsOperations() {
        List<NASSearchSpace.Operation> ops = NASSearchSpace.dartsOperations();
        assertEquals(8, ops.size());
    }

    @Test
    public void testArchitectureSpec() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        space.setArchitecture(Arrays.asList(1, 2, 3));
        
        NASSearchSpace.ArchitectureSpec spec = space.getArchitectureSpec();
        
        assertEquals("NASCell", spec.name());
        assertEquals(3, spec.numNodes());
        assertEquals(3, spec.choices().size());
        assertNotNull(spec.toCompactString());
    }

    @Test
    public void testEdgeSetSelectedOperation() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASSearchSpace.Edge edge = space.edges().get(0);
        
        edge.setSelectedOperation(2);
        assertEquals(2, edge.selectedOperationIndex());
        assertEquals(NASSearchSpace.standardOperations().get(2), edge.selectedOperation());
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testEdgeInvalidOperationIndex() {
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        NASSearchSpace.Edge edge = space.edges().get(0);
        
        edge.setSelectedOperation(100);
    }

    @Test
    public void testToString() {
        MAML maml = new MAML(new MockModule(), 0.01f, 0.001f, 5, new MSELoss());
        assertNotNull(maml.toString());
        assertTrue(maml.toString().contains("MAML"));
        
        KnowledgeDistiller distiller = new KnowledgeDistiller(
            new MockModule(), new MockModule(), 4.0f, 0.5f, new MSELoss());
        assertNotNull(distiller.toString());
        assertTrue(distiller.toString().contains("KnowledgeDistiller"));
        
        NASSearchSpace space = new NASSearchSpace(3, NASSearchSpace.standardOperations());
        assertNotNull(space.toString());
        assertTrue(space.toString().contains("NASSearchSpace"));
        
        NASController controller = new NASController(space, NASController.SearchStrategy.RANDOM);
        assertNotNull(controller.toString());
        assertTrue(controller.toString().contains("NASController"));
        
        FederatedMetaLearner learner = new FederatedMetaLearner(
            new MockModule(), new MSELoss(), 5, 10, 3, 0.01f, 0.001f);
        assertNotNull(learner.toString());
        assertTrue(learner.toString().contains("FederatedMetaLearner"));
    }

    private static class MockModule extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }
}
