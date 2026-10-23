package com.eonmux.neuralfree.acceleration.pipeline.schedule;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.*;

public class ScheduleEntryTest {

    @Test
    public void testBasicConstruction() {
        ScheduleEntry entry = new ScheduleEntry(
                "node_1",
                QueueType.COMPUTE,
                Arrays.asList("dep_1", "dep_2"),
                2
        );
        
        assertEquals("nodeId should match", "node_1", entry.nodeId());
        assertEquals("targetQueueType should match", QueueType.COMPUTE, entry.targetQueueType());
        assertEquals("waitOnNodeIds should match", Arrays.asList("dep_1", "dep_2"), entry.waitOnNodeIds());
        assertEquals("bufferSlotIndex should match", 2, entry.bufferSlotIndex());
    }

    @Test
    public void testImmutability() {
        ScheduleEntry entry = new ScheduleEntry(
                "node_1",
                QueueType.TRANSFER,
                Arrays.asList("dep_1"),
                0
        );
        
        ScheduleEntry sameEntry = new ScheduleEntry(
                "node_1",
                QueueType.COMPUTE,
                Collections.emptyList(),
                1
        );
        
        assertEquals("Entries with same nodeId should be equal", entry, sameEntry);
        assertEquals("Hash codes should match for equal entries", entry.hashCode(), sameEntry.hashCode());
    }

    @Test
    public void testEmptyWaitOnList() {
        ScheduleEntry entry = new ScheduleEntry(
                "node_1",
                QueueType.COMPUTE,
                Collections.emptyList(),
                0
        );
        
        assertTrue("waitOnNodeIds should be empty", entry.waitOnNodeIds().isEmpty());
    }

    @Test(expected = NullPointerException.class)
    public void testNullNodeIdThrows() {
        new ScheduleEntry(null, QueueType.COMPUTE, Collections.emptyList(), 0);
    }

    @Test(expected = NullPointerException.class)
    public void testNullQueueTypeThrows() {
        new ScheduleEntry("node_1", null, Collections.emptyList(), 0);
    }

    @Test(expected = NullPointerException.class)
    public void testNullWaitOnListThrows() {
        new ScheduleEntry("node_1", QueueType.COMPUTE, null, 0);
    }

    @Test
    public void testToString() {
        ScheduleEntry entry = new ScheduleEntry(
                "node_1",
                QueueType.COMPUTE,
                Arrays.asList("dep_1"),
                2
        );
        
        String str = entry.toString();
        
        assertTrue("toString should contain nodeId", str.contains("node_1"));
        assertTrue("toString should contain targetQueueType", str.contains("COMPUTE"));
        assertTrue("toString should contain bufferSlotIndex", str.contains("2"));
    }
}
