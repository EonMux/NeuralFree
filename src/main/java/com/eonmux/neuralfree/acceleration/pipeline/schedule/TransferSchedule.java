package com.eonmux.neuralfree.acceleration.pipeline.schedule;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;
import com.eonmux.neuralfree.acceleration.pipeline.ExecutionGraph;
import com.eonmux.neuralfree.acceleration.pipeline.ExecutionNode;
import com.eonmux.neuralfree.acceleration.pipeline.ExecutionNodeType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class TransferSchedule {
    private final ExecutionGraph graph;
    private final PipelineConfig config;
    private final List<ScheduleEntry> entries;
    private final Map<String, Integer> nodeToSlotIndex;
    private final int totalSlots;

    public TransferSchedule(ExecutionGraph graph, PipelineConfig config) {
        this.graph = graph;
        this.config = config;
        this.totalSlots = config.ringBufferSlotCount();
        this.entries = new ArrayList<>();
        this.nodeToSlotIndex = new HashMap<>();
        computeSchedule();
    }

    private void computeSchedule() {
        List<String> topoOrder = graph.topologicalOrder();
        Map<String, Set<String>> completedNodes = new HashMap<>();
        
        int currentComputeSlot = 0;
        int currentTransferSlot = 0;
        
        List<String> transferNodeIds = new ArrayList<>();
        List<String> computeNodeIds = new ArrayList<>();
        
        for (String nodeId : topoOrder) {
            ExecutionNode node = graph.getNode(nodeId);
            if (node.type() == ExecutionNodeType.HOST_TO_DEVICE_TRANSFER ||
                node.type() == ExecutionNodeType.DEVICE_TO_HOST_TRANSFER) {
                transferNodeIds.add(nodeId);
            } else if (node.type() == ExecutionNodeType.KERNEL_DISPATCH) {
                computeNodeIds.add(nodeId);
            }
        }

        int transferIdx = 0;
        int computeIdx = 0;
        Set<String> scheduled = new HashSet<>();
        
        for (String nodeId : topoOrder) {
            ExecutionNode node = graph.getNode(nodeId);
            QueueType queueType = determineQueueType(node);
            
            List<String> waitOn = new ArrayList<>();
            for (String depId : node.dependsOn()) {
                if (scheduled.contains(depId)) {
                    waitOn.add(depId);
                }
            }

            int slotIndex = assignSlot(node, transferIdx, computeIdx, transferNodeIds, computeNodeIds);
            
            if (node.type() == ExecutionNodeType.HOST_TO_DEVICE_TRANSFER) {
                transferIdx++;
            } else if (node.type() == ExecutionNodeType.KERNEL_DISPATCH) {
                computeIdx++;
            }

            nodeToSlotIndex.put(nodeId, slotIndex);
            
            ScheduleEntry entry = new ScheduleEntry(nodeId, queueType, waitOn, slotIndex);
            entries.add(entry);
            scheduled.add(nodeId);
        }
    }

    private QueueType determineQueueType(ExecutionNode node) {
        switch (node.type()) {
            case HOST_TO_DEVICE_TRANSFER:
            case DEVICE_TO_HOST_TRANSFER:
                return QueueType.TRANSFER;
            case KERNEL_DISPATCH:
                return QueueType.COMPUTE;
            case BARRIER:
            default:
                return QueueType.COMPUTE;
        }
    }

    private int assignSlot(ExecutionNode node, int transferIdx, int computeIdx,
                           List<String> transferNodeIds, List<String> computeNodeIds) {
        if (!config.enablePrefetch()) {
            return 0;
        }

        switch (node.type()) {
            case HOST_TO_DEVICE_TRANSFER: {
                int transferPos = transferNodeIds.indexOf(node.nodeId());
                if (transferPos == -1) {
                    return 0;
                }
                return transferPos % totalSlots;
            }
            case KERNEL_DISPATCH: {
                int computePos = computeNodeIds.indexOf(node.nodeId());
                if (computePos == -1) {
                    return 0;
                }
                return computePos % totalSlots;
            }
            case DEVICE_TO_HOST_TRANSFER:
                return (computeIdx - 1) % totalSlots;
            case BARRIER:
            default:
                return 0;
        }
    }

    public List<ScheduleEntry> entries() {
        return Collections.unmodifiableList(entries);
    }

    public ScheduleEntry getEntry(String nodeId) {
        for (ScheduleEntry entry : entries) {
            if (entry.nodeId().equals(nodeId)) {
                return entry;
            }
        }
        return null;
    }

    public int getSlotIndex(String nodeId) {
        return nodeToSlotIndex.getOrDefault(nodeId, 0);
    }

    public int totalSlots() {
        return totalSlots;
    }

    public ExecutionGraph graph() {
        return graph;
    }

    public PipelineConfig config() {
        return config;
    }

    public List<ScheduleEntry> getEntriesByQueueType(QueueType queueType) {
        List<ScheduleEntry> result = new ArrayList<>();
        for (ScheduleEntry entry : entries) {
            if (entry.targetQueueType() == queueType) {
                result.add(entry);
            }
        }
        return result;
    }

    public List<ScheduleEntry> getOverlappableTransfers(String computeNodeId) {
        ExecutionNode computeNode = graph.getNode(computeNodeId);
        if (computeNode == null || computeNode.type() != ExecutionNodeType.KERNEL_DISPATCH) {
            return Collections.emptyList();
        }

        List<ExecutionNode> overlappableNodes = graph.getOverlappableTransfers(computeNode);
        List<ScheduleEntry> result = new ArrayList<>();
        
        Set<String> scheduledBefore = new HashSet<>();
        for (ScheduleEntry entry : entries) {
            if (entry.nodeId().equals(computeNodeId)) {
                break;
            }
            scheduledBefore.add(entry.nodeId());
        }

        for (ExecutionNode node : overlappableNodes) {
            if (scheduledBefore.contains(node.nodeId())) {
                ScheduleEntry entry = getEntry(node.nodeId());
                if (entry != null) {
                    result.add(entry);
                }
            }
        }

        return result;
    }

    @Override
    public String toString() {
        return "TransferSchedule{" +
               "totalSlots=" + totalSlots +
               ", entries=" + entries.size() +
               ", config=" + config +
               '}';
    }
}
