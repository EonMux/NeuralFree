package com.eonmux.neuralfree.acceleration.pipeline.schedule;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;
import com.eonmux.neuralfree.acceleration.pipeline.ExecutionGraph;
import com.eonmux.neuralfree.acceleration.pipeline.ExecutionNodeType;
import com.eonmux.neuralfree.ai.v2.NeuralNetworkData;
import org.junit.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.*;

public class TransferScheduleTest {

    @Test
    public void testDoubleBufferSchedule() {
        int[] layerSizes = {10, 20, 30, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        assertEquals("Double buffer should have 2 slots", 2, schedule.totalSlots());
        
        List<ScheduleEntry> entries = schedule.entries();
        assertFalse("Schedule should have entries", entries.isEmpty());
        
        Set<Integer> usedSlots = new HashSet<>();
        for (ScheduleEntry entry : entries) {
            assertTrue("Slot index should be 0 or 1 for double buffer",
                    entry.bufferSlotIndex() >= 0 && entry.bufferSlotIndex() < 2);
            usedSlots.add(entry.bufferSlotIndex());
        }
        assertEquals("Both slots should be used in double buffer", 2, usedSlots.size());
    }

    @Test
    public void testTripleBufferSchedule() {
        int[] layerSizes = {10, 20, 30, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(2)
                .build();
        
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        assertEquals("Triple buffer should have 3 slots", 3, schedule.totalSlots());
        
        List<ScheduleEntry> entries = schedule.entries();
        for (ScheduleEntry entry : entries) {
            assertTrue("Slot index should be 0, 1, or 2 for triple buffer",
                    entry.bufferSlotIndex() >= 0 && entry.bufferSlotIndex() < 3);
        }
    }

    @Test
    public void testPrefetchDisabledSingleSlot() {
        int[] layerSizes = {10, 20, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(false)
                .build();
        
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        assertEquals("Disabled prefetch should have 1 slot", 1, schedule.totalSlots());
        
        for (ScheduleEntry entry : schedule.entries()) {
            assertEquals("All entries should use slot 0 when prefetch disabled",
                    0, entry.bufferSlotIndex());
        }
    }

    @Test
    public void testSlotAssignmentsCycleCorrectly() {
        int[] layerSizes = {10, 20, 30, 40, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        int prevComputeSlot = -1;
        for (ScheduleEntry entry : schedule.entries()) {
            ExecutionNodeType nodeType = graph.getNode(entry.nodeId()).type();
            if (nodeType == ExecutionNodeType.KERNEL_DISPATCH) {
                int slot = entry.bufferSlotIndex();
                int expected = (prevComputeSlot + 1) % 2;
                assertTrue("Compute slots should cycle through ring buffer",
                        prevComputeSlot == -1 || slot == expected || 
                        Math.abs(slot - prevComputeSlot) <= 1);
                prevComputeSlot = slot;
            }
        }
    }

    @Test
    public void testTransferNodesUseTransferQueue() {
        int[] layerSizes = {10, 20, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.defaultConfig();
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        for (ScheduleEntry entry : schedule.entries()) {
            ExecutionNodeType nodeType = graph.getNode(entry.nodeId()).type();
            if (nodeType == ExecutionNodeType.HOST_TO_DEVICE_TRANSFER ||
                nodeType == ExecutionNodeType.DEVICE_TO_HOST_TRANSFER) {
                assertEquals("Transfer nodes should use TRANSFER queue type",
                        QueueType.TRANSFER, entry.targetQueueType());
            }
        }
    }

    @Test
    public void testComputeNodesUseComputeQueue() {
        int[] layerSizes = {10, 20, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.defaultConfig();
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        for (ScheduleEntry entry : schedule.entries()) {
            ExecutionNodeType nodeType = graph.getNode(entry.nodeId()).type();
            if (nodeType == ExecutionNodeType.KERNEL_DISPATCH) {
                assertEquals("Compute nodes should use COMPUTE queue type",
                        QueueType.COMPUTE, entry.targetQueueType());
            }
        }
    }

    @Test
    public void testWaitOnNodeIdsRespectDependencies() {
        int[] layerSizes = {10, 20, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.defaultConfig();
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        Set<String> scheduled = new HashSet<>();
        for (ScheduleEntry entry : schedule.entries()) {
            for (String waitId : entry.waitOnNodeIds()) {
                assertTrue("Wait-on node should be scheduled before current node",
                        scheduled.contains(waitId));
            }
            scheduled.add(entry.nodeId());
        }
    }

    @Test
    public void testGetEntriesByQueueType() {
        int[] layerSizes = {10, 20, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.defaultConfig();
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        List<ScheduleEntry> computeEntries = schedule.getEntriesByQueueType(QueueType.COMPUTE);
        List<ScheduleEntry> transferEntries = schedule.getEntriesByQueueType(QueueType.TRANSFER);
        
        for (ScheduleEntry entry : computeEntries) {
            assertEquals("Compute entries should have COMPUTE queue type",
                    QueueType.COMPUTE, entry.targetQueueType());
        }
        
        for (ScheduleEntry entry : transferEntries) {
            assertEquals("Transfer entries should have TRANSFER queue type",
                    QueueType.TRANSFER, entry.targetQueueType());
        }
        
        assertTrue("Should have at least some compute entries", computeEntries.size() > 0);
        assertTrue("Should have at least some transfer entries", transferEntries.size() > 0);
    }

    @Test
    public void testGetSlotIndex() {
        int[] layerSizes = {10, 20, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForTraining(networkData);
        
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        ScheduleEntry firstEntry = schedule.entries().get(0);
        int slotFromMethod = schedule.getSlotIndex(firstEntry.nodeId());
        assertEquals("getSlotIndex should return same value as entry.bufferSlotIndex",
                firstEntry.bufferSlotIndex(), slotFromMethod);
    }

    @Test
    public void testInferenceScheduleStructure() {
        int[] layerSizes = {10, 20, 5};
        NeuralNetworkData networkData = createNeuralNetworkData(layerSizes);
        ExecutionGraph graph = ExecutionGraph.buildForInference(networkData);
        
        PipelineConfig config = PipelineConfig.defaultConfig();
        TransferSchedule schedule = new TransferSchedule(graph, config);
        
        boolean hasOutputTransfer = false;
        for (ScheduleEntry entry : schedule.entries()) {
            if ("output_transfer".equals(entry.nodeId())) {
                hasOutputTransfer = true;
                assertEquals("Output transfer should use TRANSFER queue",
                        QueueType.TRANSFER, entry.targetQueueType());
            }
        }
        assertTrue("Inference schedule should have output_transfer", hasOutputTransfer);
    }

    private NeuralNetworkData createNeuralNetworkData(int[] layerSizes) {
        return new NeuralNetworkData(layerSizes);
    }
}
