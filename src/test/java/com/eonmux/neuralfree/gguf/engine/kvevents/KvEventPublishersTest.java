package com.eonmux.neuralfree.gguf.engine.kvevents;

import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The KV cache event values and the publishers {@code neuralfree.kv.events} names: none, log, vLLM's ZMQ
 * transport (by name or as its {@code KVEventsConfig} JSON, a replica's rank applied), or a class loaded by name;
 * ranked batches stamped with their data-parallel rank, and a failing publisher losing only its batch.
 */
public class KvEventPublishersTest {

    /** A publisher the resolution loads by class name. */
    public static final class Recording implements KvEventPublisher {
        static final List<KvEventBatch> SEEN = Collections.synchronizedList(new ArrayList<>());

        @Override
        public void publish(KvEventBatch batch) {
            SEEN.add(batch);
        }
    }

    /** A class the resolution must refuse: no publisher. */
    public static final class NotAPublisher {
    }

    private static KvEventBatch batch() {
        return new KvEventBatch(1.5, Collections.singletonList(
                new KvCacheEvent.BlockRemoved(hashes(7L), KvCacheEvent.MEDIUM_CPU)), null);
    }

    /** Block hashes in the wire's integer form, as 8-byte digests. */
    static byte[][] hashes(long... values) {
        byte[][] out = new byte[values.length][];
        for (int i = 0; i < values.length; i++) {
            out[i] = hashBytes(values[i]);
        }
        return out;
    }

    /** One block hash in the wire's integer form, an 8-byte digest. */
    static byte[] hashBytes(long value) {
        byte[] out = new byte[8];
        for (int i = 0; i < 8; i++) {
            out[i] = (byte) (value >>> ((7 - i) * 8));
        }
        return out;
    }

    @Test
    public void theNamesResolveToTheirPublishers() {
        assertFalse(KvEventPublishers.named(null).enabled());
        assertFalse(KvEventPublishers.named(" none ").enabled());
        assertFalse(KvEventPublishers.named("null").enabled());
        assertTrue(KvEventPublishers.named("log").enabled());
        KvEventPublisher loaded = KvEventPublishers.named(Recording.class.getName());
        assertTrue(loaded instanceof Recording);
        assertTrue(KvEventPublishers.forConfig(MemoryConfig.builder().kvEvents(Recording.class.getName()).build())
                instanceof Recording);
        assertFalse(KvEventPublishers.forConfig(MemoryConfig.defaults()).enabled());
    }

    @Test
    public void theKvEventsConfigJsonSelectsItsPublisher() {
        String zmq = "{\"enable_kv_cache_events\": true, \"endpoint\": \"tcp://*:0\", \"topic\": \"kv\"}";
        try (KvEventPublisher publisher = KvEventPublishers.named(zmq, 2)) {
            assertTrue(publisher instanceof ZmqEventPublisher);
            assertTrue("an ephemeral port binds, got " + publisher, publisher.toString().matches("zmq:tcp://.*:\\d+")
                    && !publisher.toString().endsWith(":0"));
        }
        assertFalse(KvEventPublishers.named("{\"enable_kv_cache_events\": false}").enabled());
        assertTrue(KvEventPublishers.named("{\"enable_kv_cache_events\": true, \"publisher\": \"log\"}").enabled());
        assertTrue(KvEventPublishers.named("{\"enable_kv_cache_events\": true, \"publisher\": \""
                + Recording.class.getName() + "\"}") instanceof Recording);
        MemoryConfig memory = MemoryConfig.builder().kvEvents(zmq).build();
        assertEquals(zmq, memory.kvEvents());
        try (KvEventPublisher publisher = KvEventPublishers.forConfig(memory, 1)) {
            assertTrue(publisher instanceof ZmqEventPublisher);
        }
    }

    @Test
    public void unusableNamesRefuseByName() {
        assertTrue(assertThrows(IllegalArgumentException.class, () -> KvEventPublishers.named("com.example.Absent"))
                .getMessage().contains(MemoryConfig.PROP_KV_EVENTS));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> KvEventPublishers.named(NotAPublisher.class.getName())).getMessage().contains("KvEventPublisher"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> MemoryConfig.builder()
                .kvEvents("{\"enable_kv_cache_events\": true, \"hwm\": -1}").build()).getMessage()
                .contains(MemoryConfig.PROP_KV_EVENTS));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> KvEventPublishers.named(
                "{\"enable_kv_cache_events\": true, \"endpoint\": \"ipc:///tmp/kv\"}")).getMessage()
                .contains("endpoint"));
    }

    @Test
    public void theMemoryConfigReadsAndNormalizesTheProperty() {
        Properties p = new Properties();
        p.setProperty(MemoryConfig.PROP_KV_EVENTS, "log");
        MemoryConfig read = MemoryConfig.fromProperties(p);
        assertEquals("log", read.kvEvents());
        assertEquals("log", read.toBuilder().build().kvEvents());
        assertEquals("zmq", MemoryConfig.builder().kvEvents("zmq").build().kvEvents());
        String json = "{\"enable_kv_cache_events\": true, \"replay_endpoint\": \"tcp://*:5558\"}";
        p.setProperty(MemoryConfig.PROP_KV_EVENTS, json);
        assertEquals("the property takes the KVEventsConfig JSON", json, MemoryConfig.fromProperties(p).kvEvents());
        p.setProperty(MemoryConfig.PROP_KV_EVENTS, "{\"enable_kv_cache_events\": true, \"topik\": \"kv\"}");
        assertTrue(assertThrows(IllegalArgumentException.class, () -> MemoryConfig.fromProperties(p)).getMessage()
                .contains("topik"));
        assertNull(MemoryConfig.builder().kvEvents("none").build().kvEvents());
        assertNull(MemoryConfig.builder().kvEvents("  ").build().kvEvents());
        assertNull(MemoryConfig.defaults().kvEvents());
    }

    @Test
    public void aRankedPublisherStampsItsRankAndAListenerHearsEveryBatch() {
        List<KvEventBatch> heard = new ArrayList<>();
        KvEventPublisher ranked = KvEventPublishers.ranked(KvEventPublishers.listener(heard::add), 3);
        ranked.publish(batch());
        assertEquals(1, heard.size());
        assertEquals(Integer.valueOf(3), heard.get(0).dataParallelRank());
        assertEquals(batch().events(), heard.get(0).events());
        assertSame("a disabled publisher stays disabled", KvEventPublishers.none(),
                KvEventPublishers.ranked(KvEventPublishers.none(), 1));
        assertThrows(IllegalArgumentException.class, () -> KvEventPublishers.ranked(KvEventPublishers.log(), -1));
    }

    @Test
    public void aFailingPublisherLosesOnlyItsBatch() {
        KvEventPublisher failing = b -> {
            throw new IllegalStateException("sink down");
        };
        KvEventPublishers.publishSafely(failing, batch());
        Recording.SEEN.clear();
        KvEventPublishers.publishSafely(new Recording(), batch());
        assertEquals(1, Recording.SEEN.size());
    }

    @Test
    public void theEventsValidateAndCompareByValue() {
        List<List<BlockExtraKey>> keys = Arrays.asList(
                Arrays.asList(BlockExtraKey.media("ab12", -1), BlockExtraKey.salt("salt")), null);
        KvCacheEvent.BlockStored stored = new KvCacheEvent.BlockStored(hashes(1L, 2L), null,
                new int[] {4, 5, 6, 7}, 2, KvCacheEvent.MEDIUM_GPU, null, null, null, null, keys);
        assertEquals(stored, new KvCacheEvent.BlockStored(hashes(1L, 2L), null, new int[] {4, 5, 6, 7}, 2,
                KvCacheEvent.MEDIUM_GPU, null, null, null, null, Arrays.asList(
                        Arrays.asList(BlockExtraKey.media("ab12", -1), BlockExtraKey.salt("salt")), null)));
        assertNotEquals("the adapters name the blocks", stored, new KvCacheEvent.BlockStored(hashes(1L, 2L),
                null, new int[] {4, 5, 6, 7}, 2, KvCacheEvent.MEDIUM_GPU, 4, "sql", null, null, keys));
        assertNotEquals("the spec kind names the blocks", stored, new KvCacheEvent.BlockStored(hashes(1L, 2L),
                null, new int[] {4, 5, 6, 7}, 2, KvCacheEvent.MEDIUM_GPU, null, null, "full_attention", null, keys));
        assertArrayEquals(new int[] {4, 5, 6, 7}, stored.tokenIds());
        assertArrayEquals("the integer form reads the leading 64 bits", new long[] {1L, 2L}, stored.blockHashInts());
        assertEquals("vLLM's untagged per-block keys", "[[(ab12, -1), salt], null]",
                stored.extraKeys().toString());
        assertNotEquals("the offset is part of a media key", BlockExtraKey.media("ab12", 0),
                BlockExtraKey.media("ab12", -1));
        assertNotEquals("the kind is part of a key", BlockExtraKey.salt("ab12"), BlockExtraKey.promptEmbeds("ab12"));
        assertNotEquals("an adapter identity is no salt", BlockExtraKey.lora("ab12"), BlockExtraKey.salt("ab12"));
        assertEquals("the adapters' key publishes its bare identity", "[lora-id, (ab12, 0), salt]",
                Arrays.asList(BlockExtraKey.lora("lora-id"), BlockExtraKey.media("ab12", 0),
                        BlockExtraKey.salt("salt")).toString());
        assertThrows("the tokens must fill the blocks", IllegalArgumentException.class,
                () -> new KvCacheEvent.BlockStored(hashes(1L), null, new int[] {4, 5, 6}, 2,
                        KvCacheEvent.MEDIUM_CPU, null, null, null, null, Collections.singletonList(null)));
        assertThrows("one extra key per block", IllegalArgumentException.class,
                () -> new KvCacheEvent.BlockStored(hashes(1L), null, new int[] {4, 5}, 2,
                        KvCacheEvent.MEDIUM_CPU, null, null, null, null, Collections.emptyList()));
        assertThrows("a hash carries the 8 bytes the integer form reads", IllegalArgumentException.class,
                () -> new KvCacheEvent.BlockStored(new byte[][] {{1, 2, 3}}, null, new int[] {4}, 1,
                        KvCacheEvent.MEDIUM_CPU, null, null, null, null, Collections.singletonList(null)));
        assertThrows("a batch carries events", IllegalArgumentException.class,
                () -> new KvEventBatch(0, Collections.emptyList(), null));
        assertEquals(new KvCacheEvent.BlockRemoved(hashes(9L), KvCacheEvent.MEDIUM_CPU),
                new KvCacheEvent.BlockRemoved(hashes(9L), KvCacheEvent.MEDIUM_CPU));
    }
}
