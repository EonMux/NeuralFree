package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * The compressor rollback snapshots ({@code n_rs_seq}) {@link EngineConfig} gives the memory settings: the draft max
 * for drafters whose target llama.cpp sizes with {@code need_n_rs_seq} (MTP, eagle3, dflash; common/common.h:401-407),
 * 0 for a separate draft model or no drafter, and an explicit or pinned value kept.
 */
public class EngineConfigRollbackTest {

    @Test
    public void noDrafterKeepsNoSnapshots() {
        assertEquals(0, EngineConfig.defaults().memory().rollbackSnapshots());
        assertEquals(0, EngineConfig.builder().draftMax(6).build().memory().rollbackSnapshots());
    }

    @Test
    public void ownHeadAndMtpDraftFilesKeepTheDraftMax() {
        assertEquals(5, EngineConfig.builder().mtp(true).draftMax(5).build().memory().rollbackSnapshots());
        assertEquals(4, EngineConfig.builder().draftType(DraftType.MTP).draftMax(4).build().memory()
                .rollbackSnapshots());
        assertEquals(4, EngineConfig.builder().draftModelPath("/m/mtp.gguf").draftType(DraftType.MTP).draftMax(4)
                .build().memory().rollbackSnapshots());
        assertEquals("auto keeps the draft max until the factory resolves the file", 7,
                EngineConfig.builder().draftModelPath("/m/d.gguf").draftMax(7).build().memory().rollbackSnapshots());
    }

    @Test
    public void separateDraftModelKeepsNoSnapshots() {
        assertEquals(0, EngineConfig.builder().draftModelPath("/m/d.gguf").draftType(DraftType.MODEL).draftMax(7)
                .build().memory().rollbackSnapshots());
    }

    @Test
    public void explicitAndPinnedValuesSurviveRebuilds() {
        EngineConfig explicit = EngineConfig.builder().mtp(true).draftMax(5)
                .memory(MemoryConfig.builder().rollbackSnapshots(2).build()).build();
        assertEquals(2, explicit.memory().rollbackSnapshots());
        assertEquals(2, explicit.toBuilder().draftMax(9).build().memory().rollbackSnapshots());

        EngineConfig auto = EngineConfig.builder().draftModelPath("/m/d.gguf").draftMax(7).build();
        EngineConfig resolved = auto.toBuilder().memory(auto.memory().pinRollback(0)).build();
        assertEquals(0, resolved.memory().rollbackSnapshots());
        assertEquals(0, resolved.toBuilder().backend(EngineConfig.BackendMode.CPU).build().memory()
                .rollbackSnapshots());
        assertEquals(0, resolved.toBuilder().memory(resolved.memory().toBuilder().ubatch(64).build()).build()
                .memory().rollbackSnapshots());
    }
}
