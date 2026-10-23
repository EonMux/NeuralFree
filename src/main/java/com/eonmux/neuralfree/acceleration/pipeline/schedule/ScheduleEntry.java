package com.eonmux.neuralfree.acceleration.pipeline.schedule;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class ScheduleEntry {
    private final String nodeId;
    private final QueueType targetQueueType;
    private final List<String> waitOnNodeIds;
    private final int bufferSlotIndex;

    public ScheduleEntry(String nodeId, QueueType targetQueueType, 
                         List<String> waitOnNodeIds, int bufferSlotIndex) {
        this.nodeId = Objects.requireNonNull(nodeId, "nodeId must not be null");
        this.targetQueueType = Objects.requireNonNull(targetQueueType, "targetQueueType must not be null");
        this.waitOnNodeIds = Collections.unmodifiableList(
            new ArrayList<>(Objects.requireNonNull(waitOnNodeIds, "waitOnNodeIds must not be null"))
        );
        this.bufferSlotIndex = bufferSlotIndex;
    }

    public String nodeId() {
        return nodeId;
    }

    public QueueType targetQueueType() {
        return targetQueueType;
    }

    public List<String> waitOnNodeIds() {
        return waitOnNodeIds;
    }

    public int bufferSlotIndex() {
        return bufferSlotIndex;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ScheduleEntry that = (ScheduleEntry) o;
        return nodeId.equals(that.nodeId);
    }

    @Override
    public int hashCode() {
        return nodeId.hashCode();
    }

    @Override
    public String toString() {
        return "ScheduleEntry{" +
               "nodeId='" + nodeId + '\'' +
               ", targetQueueType=" + targetQueueType +
               ", bufferSlotIndex=" + bufferSlotIndex +
               ", waitOnNodeIds=" + waitOnNodeIds +
               '}';
    }
}
