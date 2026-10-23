package com.eonmux.neuralfree.gguf.memory;

import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Property parsing of {@link MemoryConfig}: recurrent-state snapshots and NextN draft memory follow the speculative
 * settings unless set explicitly, and {@code toBuilder} keeps every field.
 */
public class MemoryConfigTest {

    private static Properties props(String... kv) {
        Properties p = new Properties();
        for (int i = 0; i < kv.length; i += 2) {
            p.setProperty(kv[i], kv[i + 1]);
        }
        return p;
    }

    @Test
    public void noSpeculationReservesNothing() {
        MemoryConfig c = MemoryConfig.fromProperties(props());
        assertFalse(c.stateSnapshots());
        assertFalse(c.nextnMemory());
        assertFalse(MemoryConfig.defaults().stateSnapshots());
        assertFalse(MemoryConfig.defaults().nextnMemory());
    }

    @Test
    public void draftModelReservesSnapshotsOnly() {
        MemoryConfig c = MemoryConfig.fromProperties(props("neuralfree.draft", "/models/draft.gguf"));
        assertTrue(c.stateSnapshots());
        assertFalse(c.nextnMemory());
        assertFalse(MemoryConfig.fromProperties(props("neuralfree.draft", "  ")).stateSnapshots());
    }

    @Test
    public void mtpReservesSnapshotsAndNextnMemory() {
        MemoryConfig c = MemoryConfig.fromProperties(props("neuralfree.mtp", "true"));
        assertTrue(c.stateSnapshots());
        assertTrue(c.nextnMemory());
        assertFalse(MemoryConfig.fromProperties(props("neuralfree.mtp", "false")).stateSnapshots());
    }

    @Test
    public void explicitSettingsOverrideTheSpeculativeDefaults() {
        MemoryConfig c = MemoryConfig.fromProperties(props("neuralfree.mtp", "true",
                MemoryConfig.PROP_STATE_SNAPSHOTS, "false", MemoryConfig.PROP_NEXTN_MEMORY, "false"));
        assertFalse(c.stateSnapshots());
        assertFalse(c.nextnMemory());
        MemoryConfig on = MemoryConfig.fromProperties(props(MemoryConfig.PROP_STATE_SNAPSHOTS, "true"));
        assertTrue(on.stateSnapshots());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_STATE_SNAPSHOTS, "yes")));
    }

    @Test
    public void forSpeculationFollowsTheDrafterUnlessSetExplicitly() {
        MemoryConfig auto = MemoryConfig.defaults();
        MemoryConfig mtp = auto.forSpeculation(true, true);
        assertTrue(mtp.stateSnapshots());
        assertTrue(mtp.nextnMemory());
        MemoryConfig off = mtp.forSpeculation(false, false);
        assertFalse(off.stateSnapshots());
        assertFalse(off.nextnMemory());
        assertSame(auto, auto.forSpeculation(false, false));

        MemoryConfig pinned = MemoryConfig.builder().stateSnapshots(false).nextnMemory(true).build();
        MemoryConfig kept = pinned.forSpeculation(true, false);
        assertFalse(kept.stateSnapshots());
        assertTrue(kept.nextnMemory());
        MemoryConfig copied = pinned.toBuilder().ubatch(32).build().forSpeculation(true, false);
        assertFalse("toBuilder keeps an explicit setting explicit", copied.stateSnapshots());
        assertTrue(copied.nextnMemory());
    }

    @Test
    public void toBuilderKeepsTheNewSettings() {
        MemoryConfig c = MemoryConfig.builder().stateSnapshots(true).nextnMemory(true).ubatch(64).build();
        MemoryConfig copy = c.toBuilder().nCtx(1024).build();
        assertTrue(copy.stateSnapshots());
        assertTrue(copy.nextnMemory());
        assertEquals(64, copy.ubatch());
        assertEquals(1024, copy.nCtx());
    }

    @Test
    public void rollbackSnapshotsFollowTheDrafterUnlessSetOrPinned() {
        MemoryConfig auto = MemoryConfig.defaults();
        assertEquals(0, auto.rollbackSnapshots());
        assertSame(auto, auto.forRollback(0));
        MemoryConfig three = auto.forRollback(3);
        assertEquals(3, three.rollbackSnapshots());
        assertEquals(0, three.forRollback(0).rollbackSnapshots());
        assertEquals(3, three.toBuilder().ubatch(64).build().rollbackSnapshots());

        MemoryConfig pinned = auto.pinRollback(0);
        assertEquals("a pinned value survives the engine's default", 0, pinned.forRollback(5).rollbackSnapshots());
        assertEquals(0, pinned.toBuilder().nCtx(512).build().forRollback(5).rollbackSnapshots());
        assertEquals("a pin keeps an earlier explicit value", 0, pinned.pinRollback(4).rollbackSnapshots());

        MemoryConfig explicit = MemoryConfig.builder().rollbackSnapshots(2).build();
        assertEquals(2, explicit.forRollback(6).rollbackSnapshots());
        assertEquals(2, explicit.pinRollback(6).rollbackSnapshots());
        assertTrue(explicit.toString().contains("rollbackSnapshots=2"));

        assertThrows(IllegalArgumentException.class, () -> MemoryConfig.builder().rollbackSnapshots(-1).build());
        assertThrows(IllegalArgumentException.class, () -> auto.forRollback(-1));
        assertThrows(IllegalArgumentException.class, () -> auto.pinRollback(-1));
    }

    @Test
    public void draftContextUsesTheDraftCacheTypeWithoutSnapshots() {
        MemoryConfig user = MemoryConfig.builder().nCtx(8192).typeK(KvDType.Q4_0).typeV(KvDType.Q8_0).ubatch(256)
                .swaFull(true).mlaCache(MemoryConfig.MlaCacheMode.DECOMPRESSED).kvRotation(true).contextShift(true)
                .stateSnapshots(true).nextnMemory(true).build();
        MemoryConfig draft = user.forDraftContext();
        assertSame(MemoryConfig.DRAFT_KV_TYPE, draft.typeK());
        assertSame(MemoryConfig.DRAFT_KV_TYPE, draft.typeV());
        assertSame(KvDType.F16, MemoryConfig.DRAFT_KV_TYPE);
        assertFalse(draft.stateSnapshots());
        assertFalse(draft.nextnMemory());
        assertFalse(draft.contextShift());
        assertEquals(0, draft.rollbackSnapshots());
        assertEquals("draft contexts keep no compressor snapshots", 0,
                user.forRollback(4).forDraftContext().forRollback(4).rollbackSnapshots());
        assertEquals(8192, draft.nCtx());
        assertEquals(256, draft.ubatch());
        assertTrue(draft.swaFull());
        assertSame(MemoryConfig.MlaCacheMode.DECOMPRESSED, draft.mlaCache());
        assertTrue(draft.kvRotation());
        assertFalse("the draft settings stay explicit", draft.forSpeculation(true, true).stateSnapshots());
        assertEquals(draft.toString(), draft.forDraftContext().toString());
    }
}
