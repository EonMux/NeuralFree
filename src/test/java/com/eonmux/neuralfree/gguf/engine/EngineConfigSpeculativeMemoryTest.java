package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngineConfig.Builder#build()} gives the memory the speculative defaults of the draft model and MTP settings,
 * so configurations built in code reserve recurrent-state snapshots and NextN draft memory as the property path does,
 * while explicit memory settings are kept.
 */
public class EngineConfigSpeculativeMemoryTest {

    @Test
    public void noDrafterReservesNothing() {
        EngineConfig c = EngineConfig.defaults();
        assertFalse(c.memory().stateSnapshots());
        assertFalse(c.memory().nextnMemory());
    }

    @Test
    public void draftModelReservesSnapshots() {
        EngineConfig c = EngineConfig.builder().draftModelPath("/models/draft.gguf").build();
        assertTrue(c.memory().stateSnapshots());
        assertFalse(c.memory().nextnMemory());
    }

    @Test
    public void mtpReservesSnapshotsAndNextnMemory() {
        EngineConfig c = EngineConfig.builder().mtp(true).build();
        assertTrue(c.memory().stateSnapshots());
        assertTrue(c.memory().nextnMemory());
    }

    @Test
    public void removingTheDrafterDropsTheDerivedReservations() {
        EngineConfig mtp = EngineConfig.builder().mtp(true).build();
        EngineConfig plain = mtp.toBuilder().mtp(false).build();
        assertFalse(plain.memory().stateSnapshots());
        assertFalse(plain.memory().nextnMemory());
    }

    @Test
    public void explicitMemorySettingsAreKept() {
        MemoryConfig pinned = MemoryConfig.builder().stateSnapshots(false).nextnMemory(false).build();
        EngineConfig c = EngineConfig.builder().memory(pinned).mtp(true).build();
        assertFalse(c.memory().stateSnapshots());
        assertFalse(c.memory().nextnMemory());
    }
}
