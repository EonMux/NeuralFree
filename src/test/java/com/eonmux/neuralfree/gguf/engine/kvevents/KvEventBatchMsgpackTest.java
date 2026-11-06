package com.eonmux.neuralfree.gguf.engine.kvevents;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The msgpack form of the KV event batches against the bytes vLLM's own encoder writes: each expected payload is
 * what {@code msgspec.msgpack.Encoder().encode(...)} (msgspec 0.22) produces for the same {@code KVEventBatch}
 * built from vLLM's {@code vllm/distributed/kv_events.py} structs, with the {@code extra_keys}
 * ({@code to_event_extra_keys}) and {@code group_idx} the block pool sets. Block hashes ride as vLLM's integers
 * (the leading 64 bits of the digest, its default wire form) or as the raw digest bytes
 * ({@code VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES} off).
 */
public class KvEventBatchMsgpackTest {

    /**
     * {@code KVEventBatch(ts=1.5, events=[BlockStored(block_hashes=[0xfedcba9876543210, 5], parent_block_hash=None,
     * token_ids=[1, 2, 3, 4], block_size=2, lora_id=None, medium="GPU", lora_name=None, extra_keys=[None, None],
     * group_idx=0), BlockRemoved(block_hashes=[1, 0xfffffffffffffffe], medium="CPU", group_idx=0),
     * AllBlocksCleared()], data_parallel_rank=3)}.
     */
    private static final String EVERY_KIND = "93cb3ff800000000000093de000aa474797065ab426c6f636b53746f726564ac626c"
            + "6f636b5f68617368657392cffedcba987654321005b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647394"
            + "01020304aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c0aa657874"
            + "72615f6b65797392c0c0a967726f75705f6964780084a474797065ac426c6f636b52656d6f766564ac626c6f636b5f68617368"
            + "65739201cffffffffffffffffea66d656469756da3435055a967726f75705f6964780081a474797065b0416c6c426c6f636b73"
            + "436c656172656403";

    /**
     * {@code KVEventBatch(ts=1700000000.25, events=[BlockStored(block_hashes=[7, 8], parent_block_hash=6,
     * token_ids=[1, 300], block_size=1, lora_id=None, medium=None, lora_name=None, extra_keys=[("salt",), None],
     * group_idx=0)])}, no data-parallel rank.
     */
    private static final String SALTED_UNRANKED = "93cb41d954fc4010000091de000aa474797065ab426c6f636b53746f726564ac"
            + "626c6f636b5f686173686573920708b1706172656e745f626c6f636b5f6861736806a9746f6b656e5f6964739201cd012caa62"
            + "6c6f636b5f73697a6501a76c6f72615f6964c0a66d656469756dc0a96c6f72615f6e616d65c0aa65787472615f6b6579739291"
            + "a473616c74c0a967726f75705f69647800c0";

    /** The SHA-256 digest of {@code b"embeds"}, the prompt-embedding key of {@link #EVERY_EXTRA_KEY}. */
    private static final String EMBEDS_DIGEST = "804e6a0a978d599015792e2aed8279eefe7a9ca6b81f572c31da48fe2d2c0fa9";

    /**
     * {@code KVEventBatch(ts=2.0, events=[BlockStored(block_hashes=[9, 10], parent_block_hash=None,
     * token_ids=[-1, 5, 6, -1], block_size=2, lora_id=None, medium="GPU", lora_name="adapters",
     * extra_keys=[("lora-id", ("ab12", 1), "tenant", sha256(b"embeds").digest()), (("ab12", -1),)],
     * group_idx=0)], data_parallel_rank=0)}: every untagged extra key {@code to_event_extra_keys} publishes.
     */
    private static final String EVERY_EXTRA_KEY = "93cb400000000000000091de000aa474797065ab426c6f636b53746f726564ac"
            + "626c6f636b5f68617368657392090ab1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647394ff0506ffaa62"
            + "6c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65a861646170746572"
            + "73aa65787472615f6b6579739294a76c6f72612d696492a46162313201a674656e616e74c420" + EMBEDS_DIGEST
            + "9192a461623132ffa967726f75705f6964780000";

    /**
     * The pool's cache-spec metadata and the adapters' id ride: {@code BlockStored(block_hashes=[9, 10],
     * parent=None, token_ids=[-1, 5, 6, -1], block_size=2, lora_id=7, medium="GPU", lora_name="adapters",
     * extra_keys=[(...above...)], group_idx=0, kv_cache_spec_kind="sliding_window",
     * kv_cache_spec_sliding_window=512)}, rank 0 — the map opens map16 with 12 entries.
     */
    private static final String SPEC_FIELDS = "93cb400000000000000091de000ca474797065ab426c6f636b53746f726564ac626c"
            + "6f636b5f68617368657392090ab1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647394ff0506ffaa626c6f"
            + "636b5f73697a6502a76c6f72615f696407a66d656469756da3475055a96c6f72615f6e616d65a861646170746572"
            + "73aa65787472615f6b6579739294a76c6f72612d696492a46162313201a674656e616e74c420" + EMBEDS_DIGEST
            + "9192a461623132ffa967726f75705f69647800b26b765f63616368655f737065635f6b696e64ae736c6964696e675f77696e"
            + "646f77bc6b765f63616368655f737065635f736c6964696e675f77696e646f77cd020000";

    /**
     * The bytes form ({@code VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES} off): the digests ride raw, 32- and 16-byte ones
     * as msgpack bins.
     */
    private static final String BYTE_HASHES = "93cb3ff000000000000093de000aa474797065ab426c6f636b53746f726564ac626c"
            + "6f636b5f68617368657391c420000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
            + "b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f696473920102aa626c6f636b5f73697a6502a76c6f72615f69"
            + "64c0a66d656469756dc0a96c6f72615f6e616d65c0aa65787472615f6b65797391c0a967726f75705f6964780084a4747970"
            + "65ac426c6f636b52656d6f766564ac626c6f636b5f68617368657391c410101112131415161718191a1b1c1d1e1fa66d656469"
            + "756da3435055a967726f75705f6964780081a474797065b0416c6c426c6f636b73436c6561726564c0";

    @Test
    public void everyEventKindEncodesAsVllmsEncoderWritesIt() {
        KvEventBatch batch = new KvEventBatch(1.5, Arrays.asList(
                new KvCacheEvent.BlockStored(hashes(0xfedcba9876543210L, 5L), null, new int[] {1, 2, 3, 4}, 2,
                        KvCacheEvent.MEDIUM_GPU, null, null, null, null, Arrays.asList(null, null)),
                new KvCacheEvent.BlockRemoved(hashes(1L, -2L), KvCacheEvent.MEDIUM_CPU),
                KvCacheEvent.AllBlocksCleared.INSTANCE), 3);
        assertEquals(EVERY_KIND, hex(KvEventBatchMsgpack.encode(batch)));
    }

    @Test
    public void aSaltedChainWithoutARankEncodesAsVllmsEncoderWritesIt() {
        KvEventBatch batch = new KvEventBatch(1700000000.25, Arrays.asList(
                new KvCacheEvent.BlockStored(hashes(7L, 8L), hashBytes(6L), new int[] {1, 300}, 1, null, null, null,
                        null, null, Arrays.asList(Collections.singletonList(BlockExtraKey.salt("salt")), null))),
                null);
        assertEquals(SALTED_UNRANKED, hex(KvEventBatchMsgpack.encode(batch)));
    }

    @Test
    public void everyExtraKeyAndTheLoraNameEncodeAsVllmsEncoderWritesThem() {
        KvEventBatch batch = new KvEventBatch(2.0, Collections.singletonList(
                new KvCacheEvent.BlockStored(hashes(9L, 10L), null, new int[] {-1, 5, 6, -1}, 2,
                        KvCacheEvent.MEDIUM_GPU, null, "adapters", null, null, Arrays.asList(
                                Arrays.asList(BlockExtraKey.lora("lora-id"), BlockExtraKey.media("ab12", 1),
                                        BlockExtraKey.salt("tenant"), BlockExtraKey.promptEmbeds(EMBEDS_DIGEST)),
                                Collections.singletonList(BlockExtraKey.media("ab12", -1))))), 0);
        assertEquals(EVERY_EXTRA_KEY, hex(KvEventBatchMsgpack.encode(batch)));
    }

    @Test
    public void theSpecFieldsAndTheLoraIdRideBesideThem() {
        KvEventBatch batch = new KvEventBatch(2.0, Collections.singletonList(
                new KvCacheEvent.BlockStored(hashes(9L, 10L), null, new int[] {-1, 5, 6, -1}, 2,
                        KvCacheEvent.MEDIUM_GPU, 7, "adapters", "sliding_window", 512, Arrays.asList(
                                Arrays.asList(BlockExtraKey.lora("lora-id"), BlockExtraKey.media("ab12", 1),
                                        BlockExtraKey.salt("tenant"), BlockExtraKey.promptEmbeds(EMBEDS_DIGEST)),
                                Collections.singletonList(BlockExtraKey.media("ab12", -1))))), 0);
        assertEquals(SPEC_FIELDS, hex(KvEventBatchMsgpack.encode(batch)));
    }

    @Test
    public void theDigestsRideRawWhenTheIntFormIsOff() {
        byte[] wide = new byte[32];
        for (int i = 0; i < wide.length; i++) {
            wide[i] = (byte) i;
        }
        byte[] narrow = new byte[16];
        for (int i = 0; i < narrow.length; i++) {
            narrow[i] = (byte) (16 + i);
        }
        KvEventBatch batch = new KvEventBatch(1.0, Arrays.asList(
                new KvCacheEvent.BlockStored(new byte[][] {wide}, null, new int[] {1, 2}, 2, null, null, null,
                        null, null, Arrays.asList((List<BlockExtraKey>) null)),
                new KvCacheEvent.BlockRemoved(new byte[][] {narrow}, KvCacheEvent.MEDIUM_CPU),
                KvCacheEvent.AllBlocksCleared.INSTANCE), null);
        assertEquals(BYTE_HASHES, hex(KvEventBatchMsgpack.encode(batch, false)));
    }

    @Test
    public void aPromptEmbedsDigestThatIsNotHexIsRefused() {
        KvEventBatch batch = new KvEventBatch(1.0, Collections.singletonList(
                new KvCacheEvent.BlockStored(hashes(1L), null, new int[] {-1}, 1, null, null, null, null, null,
                        Collections.singletonList(Collections.singletonList(BlockExtraKey.promptEmbeds("xyz0"))))),
                null);
        try {
            KvEventBatchMsgpack.encode(batch);
            fail("a digest that is not hex has no bytes to publish");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("xyz0"));
        }
    }

    /** One block hash in the wire's integer form, an 8-byte digest. */
    private static byte[] hashBytes(long value) {
        byte[] out = new byte[8];
        for (int i = 0; i < 8; i++) {
            out[i] = (byte) (value >>> ((7 - i) * 8));
        }
        return out;
    }

    private static byte[][] hashes(long... values) {
        byte[][] out = new byte[values.length][];
        for (int i = 0; i < values.length; i++) {
            out[i] = hashBytes(values[i]);
        }
        return out;
    }

    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(Character.forDigit((b >> 4) & 0xf, 16)).append(Character.forDigit(b & 0xf, 16));
        }
        return sb.toString();
    }
}
