package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.pipeline.ExecutionNode.Operation;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;
import org.junit.Test;

import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.*;

public class ExecutionGraphTest {

    @Test
    public void testTopologicalOrderRespectsDependencies() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 30, 5), 8);

        List<String> order = graph.topologicalOrder();
        Set<String> seen = new HashSet<>();
        for (String nodeId : order) {
            ExecutionNode node = graph.getNode(nodeId);
            assertNotNull("Node should exist: " + nodeId, node);
            for (String depId : node.dependsOn()) {
                assertTrue("Dependency " + depId + " should appear before " + nodeId, seen.contains(depId));
            }
            seen.add(nodeId);
        }
        assertEquals("All nodes should be in topological order", graph.size(), order.size());
    }

    @Test
    public void testTopologicalOrderIsDeterministic() {
        NeuralNetworkData data = network(10, 20, 30, 5);
        assertEquals(ExecutionGraph.buildForTraining(data, 4).topologicalOrder(),
                ExecutionGraph.buildForTraining(data, 4).topologicalOrder());
    }

    @Test
    public void testTrainingGraphIncludesForwardAndBackward() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 5), 4);

        boolean hasForwardCompute = false;
        boolean hasBackwardCompute = false;
        boolean hasTransfer = false;
        for (String nodeId : graph.topologicalOrder()) {
            ExecutionNode node = graph.getNode(nodeId);
            if (node.type() == ExecutionNodeType.KERNEL_DISPATCH) {
                if (node.isForward()) {
                    hasForwardCompute = true;
                } else {
                    hasBackwardCompute = true;
                }
            }
            if (node.type() == ExecutionNodeType.HOST_TO_DEVICE_TRANSFER) {
                hasTransfer = true;
            }
        }
        assertTrue("Training graph should have forward compute nodes", hasForwardCompute);
        assertTrue("Training graph should have backward compute nodes", hasBackwardCompute);
        assertTrue("Training graph should have transfer nodes", hasTransfer);
    }

    @Test
    public void testInferenceGraphIsForwardOnly() {
        ExecutionGraph graph = ExecutionGraph.buildForInference(network(10, 20, 5), 4);

        for (String nodeId : graph.topologicalOrder()) {
            assertTrue("Inference graph node should be forward: " + nodeId, graph.getNode(nodeId).isForward());
        }
        List<String> order = graph.topologicalOrder();
        assertEquals("Last node should be the output transfer", "output_transfer", order.get(order.size() - 1));
        assertEquals(ExecutionNodeType.DEVICE_TO_HOST_TRANSFER, graph.getNode("output_transfer").type());
        assertEquals(Operation.READ_OUTPUT, graph.getNode("output_transfer").operation());
    }

    @Test
    public void testInferenceGraphUsesDenseKernels() {
        ExecutionGraph graph = ExecutionGraph.buildForInference(network(10, 20, 5), 4);

        Map<Operation, Integer> counts = countOperations(graph);
        assertEquals(Integer.valueOf(2), counts.get(Operation.LINEAR));
        assertEquals(Integer.valueOf(2), counts.get(Operation.BIAS_ACTIVATION));
        for (ExecutionNode node : graph.getNodesByType(ExecutionNodeType.KERNEL_DISPATCH)) {
            String expected = node.operation() == Operation.LINEAR ? Kernels.GEMM_F32 : Kernels.DENSE_BIAS_ACT_F32;
            assertEquals("kernel of " + node.nodeId(), expected, node.kernelName());
        }
        assertNull(counts.get(Operation.SGD_UPDATE));
        assertNull(counts.get(Operation.UPLOAD_TARGET));
    }

    @Test
    public void testTrainingGraphNamesTrainKernels() {
        int layers = 3;
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 30, 5), 4);

        for (ExecutionNode node : graph.getNodesByType(ExecutionNodeType.KERNEL_DISPATCH)) {
            assertTrue("kernel " + node.kernelName() + " of " + node.nodeId() + " is a train kernel",
                    Kernels.ALL.contains(node.kernelName()));
            assertEquals("kernel of " + node.nodeId(), expectedKernel(node.operation()), node.kernelName());
        }
        Map<Operation, Integer> counts = countOperations(graph);
        assertEquals(Integer.valueOf(layers), counts.get(Operation.LINEAR));
        assertEquals(Integer.valueOf(layers), counts.get(Operation.BIAS_ACTIVATION));
        assertEquals(Integer.valueOf(1), counts.get(Operation.LOSS_GRADIENT));
        assertEquals(Integer.valueOf(layers), counts.get(Operation.ACTIVATION_BACKWARD));
        assertEquals(Integer.valueOf(layers), counts.get(Operation.WEIGHT_GRADIENT));
        assertEquals(Integer.valueOf(layers), counts.get(Operation.BIAS_GRADIENT));
        assertEquals(Integer.valueOf(layers - 1), counts.get(Operation.INPUT_GRADIENT));
        assertEquals(Integer.valueOf(1), counts.get(Operation.SGD_UPDATE));
        assertEquals(Integer.valueOf(1), counts.get(Operation.READ_RESULTS));
        assertEquals(Integer.valueOf(1), counts.get(Operation.READ_OUTPUT));
        List<String> order = graph.topologicalOrder();
        assertTrue("the loss gradient follows the last forward layer",
                order.indexOf("backward_loss_grad") > order.indexOf("forward_compute_L" + (layers - 1)));
    }

    @Test
    public void testGradientGraphHasNoUpdate() {
        ExecutionGraph graph = ExecutionGraph.buildForGradients(network(10, 20, 5), 4);

        Map<Operation, Integer> counts = countOperations(graph);
        assertNull(counts.get(Operation.SGD_UPDATE));
        assertNull(counts.get(Operation.READ_OUTPUT));
        assertEquals(Integer.valueOf(1), counts.get(Operation.READ_RESULTS));
        List<String> order = graph.topologicalOrder();
        assertEquals("result_transfer", order.get(order.size() - 1));
    }

    @Test
    public void testInputGradientWaitsForTheGradientsItOverwrites() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(6, 7, 8, 9, 3), 5);
        List<String> order = graph.topologicalOrder();

        for (int layer = 1; layer < 3; layer++) {
            int inputGrad = order.indexOf("backward_input_grad_L" + layer);
            assertTrue(inputGrad > order.indexOf("backward_compute_grad_L" + (layer + 1)));
            assertTrue(inputGrad > order.indexOf("backward_bias_grad_L" + (layer + 1)));
            assertTrue(inputGrad > order.indexOf("backward_act_L" + layer));
        }
    }

    @Test
    public void testUpdateRunsAfterEveryWeightReader() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(6, 7, 8, 3), 5);
        List<String> order = graph.topologicalOrder();
        int update = order.indexOf("update_weights");

        assertTrue(update >= 0);
        for (String id : order) {
            Operation op = graph.getNode(id).operation();
            if (op == Operation.LINEAR || op == Operation.WEIGHT_GRADIENT || op == Operation.BIAS_GRADIENT
                    || op == Operation.INPUT_GRADIENT) {
                assertTrue(id + " runs before the update", order.indexOf(id) < update);
            }
        }
        assertTrue(order.indexOf("result_transfer") > update);
        assertTrue(order.indexOf("output_transfer") > update);
    }

    @Test
    public void testUploadsPrecedeEveryKernelAndReadbacksFollowThem() {
        for (ExecutionGraph graph : new ExecutionGraph[]{
                ExecutionGraph.buildForTraining(network(6, 7, 8, 3), 5),
                ExecutionGraph.buildForGradients(network(6, 7, 8, 3), 5),
                ExecutionGraph.buildForInference(network(6, 7, 8, 3), 5)}) {
            List<String> order = graph.topologicalOrder();
            int lastUpload = -1;
            int firstKernel = order.size();
            int lastKernel = -1;
            int firstReadback = order.size();
            for (int i = 0; i < order.size(); i++) {
                ExecutionNodeType type = graph.getNode(order.get(i)).type();
                if (type == ExecutionNodeType.HOST_TO_DEVICE_TRANSFER) {
                    lastUpload = i;
                } else if (type == ExecutionNodeType.KERNEL_DISPATCH) {
                    firstKernel = Math.min(firstKernel, i);
                    lastKernel = i;
                } else if (type == ExecutionNodeType.DEVICE_TO_HOST_TRANSFER) {
                    firstReadback = Math.min(firstReadback, i);
                }
            }
            assertTrue("every upload precedes the first kernel: " + order, lastUpload < firstKernel);
            assertTrue("every readback follows the last kernel: " + order, firstReadback > lastKernel);
        }
    }

    @Test
    public void testGetOverlappableTransfersReturnsNextLayerTransfer() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 30, 5), 4);

        List<ExecutionNode> overlappable = graph.getOverlappableTransfers(graph.getNode("forward_compute_L0"));
        boolean hasLayer1Transfer = false;
        for (ExecutionNode node : overlappable) {
            if (node.layerIndex() == 1 && node.isForward()) {
                hasLayer1Transfer = true;
            }
        }
        assertTrue("Layer 0 compute should be able to overlap with layer 1 transfer", hasLayer1Transfer);
    }

    @Test
    public void testGetOverlappableTransfersEmptyForLastLayer() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 5), 4);

        List<ExecutionNode> overlappable = graph.getOverlappableTransfers(graph.getNode("forward_compute_L1"));
        assertTrue("Last layer compute should have no overlappable transfers ahead", overlappable.isEmpty());
    }

    @Test
    public void testGetOverlappableTransfersEmptyForNonComputeNodes() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 5), 4);

        List<ExecutionNode> overlappable =
                graph.getOverlappableTransfers(graph.getNode("forward_transfer_weights_L0"));
        assertTrue("Non-compute nodes should have empty overlappable transfers", overlappable.isEmpty());
    }

    @Test
    public void testLayerIndexSetCorrectly() {
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 30, 5), 4);

        for (int layer = 0; layer < 3; layer++) {
            assertEquals(layer, graph.getNode("forward_compute_L" + layer).layerIndex());
            assertEquals(layer, graph.getNode("forward_linear_L" + layer).layerIndex());
            assertEquals(layer, graph.getNode("forward_transfer_weights_L" + layer).layerIndex());
            assertEquals(layer, graph.getNode("backward_act_L" + layer).layerIndex());
        }
        assertEquals(-1, graph.getNode("transfer_input").layerIndex());
    }

    @Test
    public void testTransferSizeCalculatedCorrectly() {
        int rows = 3;
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 5), rows);

        assertEquals((long) 10 * 20 * Float.BYTES, graph.getNode("forward_transfer_weights_L0").transferSizeBytes());
        assertEquals(20L * Float.BYTES, graph.getNode("forward_transfer_biases_L0").transferSizeBytes());
        assertEquals((long) rows * 10 * Float.BYTES, graph.getNode("transfer_input").transferSizeBytes());
        assertEquals((long) rows * 5 * Float.BYTES, graph.getNode("transfer_target").transferSizeBytes());
        long params = 10 * 20 + 20 + 20 * 5 + 5;
        assertEquals(2 * params * Float.BYTES, graph.getNode("result_transfer").transferSizeBytes());
    }

    @Test
    public void testNodeImmutability() {
        ExecutionNode node = ExecutionNode.builder()
                .nodeId("test_node")
                .label("Test Node")
                .type(ExecutionNodeType.KERNEL_DISPATCH)
                .layerIndex(0)
                .kernelName("test_kernel")
                .globalWorkSize(new long[]{100, 1, 1})
                .localWorkSize(new long[]{32, 1, 1})
                .build();

        long[] globalSize = node.globalWorkSize();
        globalSize[0] = 999;

        assertNotEquals("Modifying returned array should not affect node", 999, node.globalWorkSize()[0]);
        assertNull("a node built without an operation has none", node.operation());
    }

    @Test
    public void testWorkSizesSetCorrectly() {
        int rows = 6;
        ExecutionGraph graph = ExecutionGraph.buildForTraining(network(10, 20, 5), rows);

        ExecutionNode compute0 = graph.getNode("forward_compute_L0");
        assertEquals("Global work size should be outputSize * rows", 20L * rows, compute0.globalWorkSize()[0]);
        assertEquals(256L, compute0.localWorkSize()[0]);
        assertEquals(5L * rows, graph.getNode("backward_loss_grad").globalWorkSize()[0]);
        assertEquals(10L * 20 + 20 + 20 * 5 + 5, graph.getNode("update_weights").globalWorkSize()[0]);
        assertNull("GEMM geometry is chosen by the dispatcher", graph.getNode("forward_linear_L0").globalWorkSize());
        assertEquals(rows, graph.batchRows());
    }

    @Test
    public void testDefaultRowsAreTheBatchSize() {
        NeuralNetworkData data = network(10, 20, 5);
        data.setBatchSize(7);

        ExecutionGraph graph = ExecutionGraph.buildForTraining(data);

        assertEquals(7, graph.batchRows());
        assertEquals(20L * 7, graph.getNode("forward_compute_L0").globalWorkSize()[0]);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testZeroRowsRejected() {
        ExecutionGraph.buildForInference(network(10, 5), 0);
    }

    private static Map<Operation, Integer> countOperations(ExecutionGraph graph) {
        Map<Operation, Integer> counts = new EnumMap<>(Operation.class);
        for (String id : graph.nodeIds()) {
            Operation op = graph.getNode(id).operation();
            assertNotNull("node " + id + " has an operation", op);
            counts.merge(op, 1, Integer::sum);
        }
        return counts;
    }

    private static String expectedKernel(Operation op) {
        switch (op) {
            case LINEAR:
            case WEIGHT_GRADIENT:
            case INPUT_GRADIENT:
                return Kernels.GEMM_F32;
            case BIAS_ACTIVATION:
                return Kernels.DENSE_BIAS_ACT_F32;
            case LOSS_GRADIENT:
                return Kernels.AXPBY_F32;
            case ACTIVATION_BACKWARD:
                return Kernels.DENSE_ACT_BACKWARD_F32;
            case BIAS_GRADIENT:
                return Kernels.REDUCE_COLS_F32;
            case SGD_UPDATE:
                return Kernels.SGD_STEP_F32;
            default:
                throw new AssertionError("not a kernel operation: " + op);
        }
    }

    private static NeuralNetworkData network(int... layerSizes) {
        return new NeuralNetworkData(layerSizes);
    }
}
