package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.engine.batch.PrefixHashAlgo;
import com.eonmux.neuralfree.gguf.engine.batch.SchedulingPolicy;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Property parsing of {@link MemoryConfig}: recurrent-state snapshots and NextN draft memory follow the speculative
 * settings unless set explicitly, the draft-context cache types and ubatch parse from their properties, the
 * scheduling policy parses by name, and {@code toBuilder} keeps every field.
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
    public void chunkedPrefillDefaultsOnWithTheUbatchAsTheBudgetAndNoCap() {
        MemoryConfig c = MemoryConfig.defaults();
        assertTrue(c.chunkedPrefill());
        assertEquals(0, c.maxBatchedTokens());
        assertEquals(0, c.longPrefillTokens());
        assertEquals(0, c.maxEncoderTokens());
        assertEquals(0, c.encoderCacheRows());
    }

    @Test
    public void theStepBudgetAndLongPrefillCapParseAndValidate() {
        MemoryConfig c = MemoryConfig.fromProperties(props(
                MemoryConfig.PROP_CHUNKED_PREFILL, "false",
                MemoryConfig.PROP_MAX_BATCHED_TOKENS, "2048",
                MemoryConfig.PROP_LONG_PREFILL_TOKENS, "512"));
        assertFalse(c.chunkedPrefill());
        assertEquals(2048, c.maxBatchedTokens());
        assertEquals(512, c.longPrefillTokens());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_MAX_BATCHED_TOKENS, "-1")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_LONG_PREFILL_TOKENS, "-8")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_MAX_BATCHED_TOKENS, "many")));
    }

    @Test
    public void theEncoderBudgetAndCacheRowsParseAndValidate() {
        MemoryConfig c = MemoryConfig.fromProperties(props(
                MemoryConfig.PROP_MAX_ENCODER_TOKENS, "4096",
                MemoryConfig.PROP_ENCODER_CACHE_ROWS, "8192"));
        assertEquals(4096, c.maxEncoderTokens());
        assertEquals(8192, c.encoderCacheRows());
        assertEquals(-1, MemoryConfig.fromProperties(props(MemoryConfig.PROP_ENCODER_CACHE_ROWS, "-1"))
                .encoderCacheRows());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_MAX_ENCODER_TOKENS, "-1")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_ENCODER_CACHE_ROWS, "-2")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_ENCODER_CACHE_ROWS, "soon")));
    }

    @Test
    public void toBuilderKeepsTheSchedulingSettings() {
        MemoryConfig c = MemoryConfig.builder().chunkedPrefill(false).maxBatchedTokens(9).longPrefillTokens(3)
                .maxEncoderTokens(7).encoderCacheRows(5).schedulingPolicy(SchedulingPolicy.PRIORITY)
                .build().toBuilder().build();
        assertFalse(c.chunkedPrefill());
        assertEquals(9, c.maxBatchedTokens());
        assertEquals(3, c.longPrefillTokens());
        assertEquals(7, c.maxEncoderTokens());
        assertEquals(5, c.encoderCacheRows());
        assertEquals(SchedulingPolicy.PRIORITY, c.schedulingPolicy());
    }

    @Test
    public void theSchedulingPolicyDefaultsToFcfsAndParsesByName() {
        assertEquals(SchedulingPolicy.FCFS, MemoryConfig.defaults().schedulingPolicy());
        assertEquals(SchedulingPolicy.FCFS, MemoryConfig.fromProperties(props()).schedulingPolicy());
        assertEquals(SchedulingPolicy.PRIORITY, MemoryConfig.fromProperties(
                props(MemoryConfig.PROP_SCHEDULING_POLICY, " Priority ")).schedulingPolicy());
        assertEquals(SchedulingPolicy.FCFS, MemoryConfig.fromProperties(
                props(MemoryConfig.PROP_SCHEDULING_POLICY, "fcfs")).schedulingPolicy());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_SCHEDULING_POLICY, "lifo")));
        assertTrue(e.getMessage(), e.getMessage().contains(MemoryConfig.PROP_SCHEDULING_POLICY));
        assertTrue(MemoryConfig.builder().schedulingPolicy(SchedulingPolicy.PRIORITY).build().toString()
                .contains("schedulingPolicy=priority"));
        assertFalse(MemoryConfig.defaults().toString().contains("schedulingPolicy"));
    }

    @Test
    public void thePrefixHashAlgoDefaultsToSha256AndParsesByName() {
        assertEquals(PrefixHashAlgo.SHA256, MemoryConfig.defaults().prefixHashAlgo());
        assertEquals(PrefixHashAlgo.SHA256, MemoryConfig.fromProperties(props()).prefixHashAlgo());
        assertEquals(PrefixHashAlgo.SHA256_CBOR, MemoryConfig.fromProperties(
                props(MemoryConfig.PROP_PREFIX_HASH_ALGO, " SHA256_CBOR ")).prefixHashAlgo());
        assertEquals(PrefixHashAlgo.XXHASH, MemoryConfig.fromProperties(
                props(MemoryConfig.PROP_PREFIX_HASH_ALGO, "xxhash")).prefixHashAlgo());
        assertEquals(PrefixHashAlgo.XXHASH_CBOR, MemoryConfig.fromProperties(
                props(MemoryConfig.PROP_PREFIX_HASH_ALGO, "xxhash_cbor")).prefixHashAlgo());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_PREFIX_HASH_ALGO, "murmur")));
        assertTrue(e.getMessage(), e.getMessage().contains(MemoryConfig.PROP_PREFIX_HASH_ALGO));
        assertTrue(e.getMessage(), e.getMessage().contains("sha256_cbor"));
        assertEquals("the builder round trips", PrefixHashAlgo.SHA256_CBOR,
                MemoryConfig.builder().prefixHashAlgo(PrefixHashAlgo.SHA256_CBOR)
                        .build().toBuilder().build().prefixHashAlgo());
        assertTrue(MemoryConfig.builder().prefixHashAlgo(PrefixHashAlgo.XXHASH).build().toString()
                .contains("prefixHashAlgo=xxhash"));
        assertFalse(MemoryConfig.defaults().toString().contains("prefixHashAlgo"));
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
    public void prefixReuseIsExplicitOnlyWhenSet() {
        assertFalse(MemoryConfig.defaults().prefixReuseSet());
        assertTrue(MemoryConfig.defaults().prefixReuse());
        assertFalse(MemoryConfig.fromProperties(props()).prefixReuseSet());
        assertTrue(MemoryConfig.fromProperties(props(MemoryConfig.PROP_PREFIX_REUSE, "true")).prefixReuseSet());
        assertFalse(MemoryConfig.fromProperties(props(MemoryConfig.PROP_PREFIX_REUSE, "false")).prefixReuse());
        assertTrue(MemoryConfig.fromProperties(props(MemoryConfig.PROP_PREFIX_REUSE, "false")).prefixReuseSet());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_PREFIX_REUSE, "yes")));
    }

    @Test
    public void contextCheckpointSettingsParseAndValidate() {
        MemoryConfig defaults = MemoryConfig.defaults();
        assertEquals(MemoryConfig.DEFAULT_CTX_CHECKPOINTS, defaults.ctxCheckpoints());
        assertEquals(MemoryConfig.DEFAULT_CHECKPOINT_MIN_STEP, defaults.checkpointMinStep());
        MemoryConfig c = MemoryConfig.fromProperties(props(MemoryConfig.PROP_CTX_CHECKPOINTS, "4",
                MemoryConfig.PROP_CHECKPOINT_MIN_STEP, "0"));
        assertEquals(4, c.ctxCheckpoints());
        assertEquals(0, c.checkpointMinStep());
        assertEquals("toBuilder keeps both", c.ctxCheckpoints(), c.toBuilder().ubatch(32).build().ctxCheckpoints());
        assertEquals(0, c.toBuilder().ubatch(32).build().checkpointMinStep());
        assertTrue(c.toString(), c.toString().contains("ctxCheckpoints=4"));
        assertEquals(0, MemoryConfig.fromProperties(props(MemoryConfig.PROP_CTX_CHECKPOINTS, "0")).ctxCheckpoints());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_CTX_CHECKPOINTS, "-1")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_CHECKPOINT_MIN_STEP, "-8")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_CTX_CHECKPOINTS, "some")));
    }

    @Test
    public void swaCheckpointReserveParsesAndValidates() {
        assertEquals(0, MemoryConfig.defaults().swaCheckpointReserve());
        MemoryConfig c = MemoryConfig.fromProperties(props(MemoryConfig.PROP_SWA_CHECKPOINT_RESERVE, "12"));
        assertEquals(12, c.swaCheckpointReserve());
        assertEquals("toBuilder keeps it", 12, c.toBuilder().ubatch(32).build().swaCheckpointReserve());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_SWA_CHECKPOINT_RESERVE, "-2")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.builder().swaCheckpointReserve(-1).build());
    }

    @Test
    public void toBuilderKeepsPrefixReuseExplicitness() {
        MemoryConfig explicit = MemoryConfig.builder().prefixReuse(false).build();
        assertTrue(explicit.prefixReuseSet());
        assertFalse(explicit.toBuilder().ubatch(32).build().prefixReuse());
        assertTrue("toBuilder keeps an explicit setting explicit",
                explicit.toBuilder().ubatch(32).build().prefixReuseSet());
        MemoryConfig auto = MemoryConfig.defaults();
        assertFalse(auto.toBuilder().kvPaged(true).build().prefixReuseSet());
        assertTrue(auto.toBuilder().kvPaged(true).build().prefixReuse());
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

    @Test
    public void draftCacheTypesParseFromProperties() {
        MemoryConfig c = MemoryConfig.fromProperties(props(MemoryConfig.PROP_DRAFT_KV_TYPE, "q8_0"));
        assertSame(KvDType.Q8_0, c.draftTypeK());
        assertSame(KvDType.Q8_0, c.draftTypeV());
        assertSame("the main cache types stay untouched", KvDType.F16, c.typeK());
        MemoryConfig split = MemoryConfig.fromProperties(props(
                MemoryConfig.PROP_DRAFT_KV_TYPE, "q8_0",
                MemoryConfig.PROP_DRAFT_KV_TYPE_K, "q4_0",
                MemoryConfig.PROP_DRAFT_KV_TYPE_V, "q5_1"));
        assertSame(KvDType.Q4_0, split.draftTypeK());
        assertSame(KvDType.Q5_1, split.draftTypeV());
        assertSame(KvDType.F16, MemoryConfig.defaults().draftTypeK());
        assertSame(KvDType.F16, MemoryConfig.defaults().draftTypeV());
        for (String key : new String[]{MemoryConfig.PROP_DRAFT_KV_TYPE, MemoryConfig.PROP_DRAFT_KV_TYPE_K,
                MemoryConfig.PROP_DRAFT_KV_TYPE_V}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> MemoryConfig.fromProperties(props(key, "q6_k")));
            assertTrue(e.getMessage(), e.getMessage().contains(key));
        }
    }

    @Test
    public void draftUbatchParsesFromProperties() {
        assertEquals(256, MemoryConfig.fromProperties(props(MemoryConfig.PROP_DRAFT_UBATCH, "256")).draftUbatch());
        assertEquals("0 keeps the target's ubatch", 0,
                MemoryConfig.fromProperties(props(MemoryConfig.PROP_DRAFT_UBATCH, "0")).draftUbatch());
        assertEquals(0, MemoryConfig.defaults().draftUbatch());
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_DRAFT_UBATCH, "-1")));
        assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(props(MemoryConfig.PROP_DRAFT_UBATCH, "half")));
    }

    @Test
    public void draftContextUsesTheConfiguredDraftCacheTypesAndUbatch() {
        MemoryConfig user = MemoryConfig.builder().nCtx(8192).typeK(KvDType.Q4_0).typeV(KvDType.Q4_0).ubatch(512)
                .draftTypeK(KvDType.Q8_0).draftTypeV(KvDType.Q4_0).draftUbatch(128).build();
        MemoryConfig draft = user.forDraftContext();
        assertSame(KvDType.Q8_0, draft.typeK());
        assertSame(KvDType.Q4_0, draft.typeV());
        assertSame(KvDType.Q8_0, draft.draftTypeK());
        assertSame(KvDType.Q4_0, draft.draftTypeV());
        assertEquals(128, draft.draftUbatch());
        assertEquals("the draft ubatch replaces the target plan's", 128, draft.draftUbatchOr(512));
        assertEquals("no draft ubatch keeps the target plan's", 512, MemoryConfig.defaults().draftUbatchOr(512));
        assertEquals(draft.toString(), draft.forDraftContext().toString());
        assertTrue(draft.toString().contains("draftK=Q8_0"));
        assertTrue(draft.toString().contains("draftUbatch=128"));

        MemoryConfig copy = draft.toBuilder().nCtx(1024).build();
        assertSame(KvDType.Q8_0, copy.draftTypeK());
        assertSame(KvDType.Q4_0, copy.draftTypeV());
        assertEquals(128, copy.draftUbatch());
        assertEquals(1024, copy.nCtx());
        assertThrows(IllegalArgumentException.class, () -> MemoryConfig.builder().draftUbatch(-1).build());
    }
}
