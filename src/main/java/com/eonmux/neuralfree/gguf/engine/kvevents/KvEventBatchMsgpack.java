package com.eonmux.neuralfree.gguf.engine.kvevents;

import org.msgpack.core.MessageBufferPacker;
import org.msgpack.core.MessagePack;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.math.BigInteger;
import java.util.List;

/**
 * The msgpack form of a {@link KvEventBatch}, byte for byte what vLLM's {@code msgspec.msgpack.Encoder} writes for
 * the {@code KVEventBatch} its block pool publishes: the batch is the array {@code [ts, events, data_parallel_rank]}
 * (the rank nil when unset), and every event is the map {@code {"type": <class name>, ...fields}} in declaration
 * order, the fields whose value is their None default left out. {@code lora_id}, {@code medium} and
 * {@code lora_name} have no default, so they always ride along; {@code extra_keys} carries one entry per block in
 * vLLM's untagged {@code to_event_extra_keys} form, nil for a block without any, and {@code group_idx} names the
 * engine's one KV cache group, 0; {@code kv_cache_spec_kind} and {@code kv_cache_spec_sliding_window} ride when the
 * pool knows them. The {@code BlockStored} map always opens as map16 — msgspec sizes the header from the struct's
 * declared fields, 16 with the tag — with the count of the entries actually written.
 *
 * <p>Block hashes ride in vLLM's wire form: the leading 64 bits of each digest as an unsigned integer (the
 * default, vLLM's {@code VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES} on), or the raw digest bytes when the environment
 * variable is set to 0.
 */
final class KvEventBatchMsgpack {

    /** The event tag map key, msgspec's {@code tag=True} spelling. */
    private static final String TYPE = "type";

    /** The msgpack map16 marker: {@code BlockStored} declares more fields than a fixmap holds, so msgspec writes the
     * map16 header even when the omitted fields leave fewer entries. */
    private static final byte MAP16 = (byte) 0xde;

    /** The KV cache group every event names: the engine keeps all of a block's layers in one group. */
    private static final int GROUP_IDX = 0;

    private static final BigInteger UNSIGNED_LONG_MASK = BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE);

    private KvEventBatchMsgpack() {
    }

    /** Whether block hashes ride as integers: vLLM's {@code VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES}, on by default. */
    static boolean intHashesByDefault() {
        String env = System.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES");
        if (env == null) {
            return true;
        }
        try {
            return Integer.parseInt(env.trim()) != 0;
        } catch (NumberFormatException e) {
            return true;
        }
    }

    /** The msgpack bytes a vLLM publisher sends for {@code batch}, block hashes as vLLM's integers. */
    static byte[] encode(KvEventBatch batch) {
        return encode(batch, intHashesByDefault());
    }

    /** The msgpack bytes a vLLM publisher sends for {@code batch}; {@code intHashes} picks the block-hash wire form. */
    static byte[] encode(KvEventBatch batch, boolean intHashes) {
        try (MessageBufferPacker packer = MessagePack.newDefaultBufferPacker()) {
            packer.packArrayHeader(3);
            packer.packDouble(batch.ts());
            List<KvCacheEvent> events = batch.events();
            packer.packArrayHeader(events.size());
            for (KvCacheEvent event : events) {
                encodeEvent(packer, event, intHashes);
            }
            Integer rank = batch.dataParallelRank();
            if (rank == null) {
                packer.packNil();
            } else {
                packer.packInt(rank);
            }
            return packer.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException("encoding the KV event batch failed", e);
        }
    }

    private static void encodeEvent(MessageBufferPacker packer, KvCacheEvent event, boolean intHashes)
            throws IOException {
        if (event instanceof KvCacheEvent.BlockStored) {
            encodeBlockStored(packer, (KvCacheEvent.BlockStored) event, intHashes);
        } else if (event instanceof KvCacheEvent.BlockRemoved) {
            encodeBlockRemoved(packer, (KvCacheEvent.BlockRemoved) event, intHashes);
        } else if (event instanceof KvCacheEvent.AllBlocksCleared) {
            packer.packMapHeader(1);
            packer.packString(TYPE);
            packer.packString("AllBlocksCleared");
        } else {
            throw new IllegalStateException("no msgpack form for the KV cache event " + event.getClass().getName());
        }
    }

    /**
     * {@code {"type": "BlockStored", "block_hashes": [...], "parent_block_hash": ..., "token_ids": [...],
     * "block_size": ..., "lora_id": ..., "medium": ..., "lora_name": ..., "extra_keys": [...], "group_idx": 0,
     * "kv_cache_spec_kind": ..., "kv_cache_spec_sliding_window": ...}}, the optional keys omitted when unset.
     * {@code extra_keys} always rides, one nil-or-tuple entry per block, as vLLM's block pool passes the per-block
     * list for every store.
     */
    private static void encodeBlockStored(MessageBufferPacker packer, KvCacheEvent.BlockStored stored,
                                          boolean intHashes) throws IOException {
        byte[][] hashes = stored.blockHashes();
        int entries = 10; // the 8 always-present entries above plus extra_keys and group_idx
        if (stored.kvCacheSpecKind() != null) {
            entries++;
        }
        if (stored.kvCacheSpecSlidingWindow() != null) {
            entries++;
        }
        packer.addPayload(new byte[] {MAP16, (byte) (entries >>> 8), (byte) entries});
        packer.packString(TYPE);
        packer.packString("BlockStored");
        packer.packString("block_hashes");
        packHashes(packer, hashes, intHashes);
        packer.packString("parent_block_hash");
        if (stored.parentBlockHash() == null) {
            packer.packNil();
        } else {
            packHash(packer, stored.parentBlockHash(), intHashes);
        }
        packer.packString("token_ids");
        int[] tokenIds = stored.tokenIds();
        packer.packArrayHeader(tokenIds.length);
        for (int token : tokenIds) {
            packer.packInt(token);
        }
        packer.packString("block_size");
        packer.packInt(stored.blockSize());
        packer.packString("lora_id");
        if (stored.loraId() == null) {
            packer.packNil();
        } else {
            packer.packInt(stored.loraId());
        }
        packer.packString("medium");
        packNullableString(packer, stored.medium());
        packer.packString("lora_name");
        packNullableString(packer, stored.loraName());
        packer.packString("extra_keys");
        packExtraKeys(packer, stored.extraKeys());
        packer.packString("group_idx");
        packer.packInt(GROUP_IDX);
        if (stored.kvCacheSpecKind() != null) {
            packer.packString("kv_cache_spec_kind");
            packer.packString(stored.kvCacheSpecKind());
        }
        if (stored.kvCacheSpecSlidingWindow() != null) {
            packer.packString("kv_cache_spec_sliding_window");
            packer.packInt(stored.kvCacheSpecSlidingWindow());
        }
    }

    /** {@code {"type": "BlockRemoved", "block_hashes": [...], "medium": ..., "group_idx": 0}}. */
    private static void encodeBlockRemoved(MessageBufferPacker packer, KvCacheEvent.BlockRemoved removed,
                                           boolean intHashes) throws IOException {
        packer.packMapHeader(4);
        packer.packString(TYPE);
        packer.packString("BlockRemoved");
        packer.packString("block_hashes");
        packHashes(packer, removed.blockHashes(), intHashes);
        packer.packString("medium");
        packNullableString(packer, removed.medium());
        packer.packString("group_idx");
        packer.packInt(GROUP_IDX);
    }

    /** One entry per block: the tuple of the block's untagged extra keys, or nil for a block without any. */
    private static void packExtraKeys(MessageBufferPacker packer, List<List<BlockExtraKey>> extraKeys)
            throws IOException {
        packer.packArrayHeader(extraKeys.size());
        for (List<BlockExtraKey> keys : extraKeys) {
            if (keys == null || keys.isEmpty()) {
                packer.packNil();
                continue;
            }
            packer.packArrayHeader(keys.size());
            for (BlockExtraKey key : keys) {
                packExtraKey(packer, key);
            }
        }
    }

    /**
     * One untagged extra key: the {@code (mm_identifier, offset)} pair of a media item, the raw SHA-256 bytes of a
     * prompt-embedding digest (vLLM hashes to {@code bytes}, which msgpack writes as bin), and the bare string of
     * the adapters' identity or the cache salt.
     */
    private static void packExtraKey(MessageBufferPacker packer, BlockExtraKey key) throws IOException {
        switch (key.kind()) {
            case MEDIA:
                packer.packArrayHeader(2);
                packer.packString(key.value());
                packer.packInt(key.offset());
                break;
            case PROMPT_EMBEDS:
                byte[] digest = hexBytes(key.value());
                packer.packBinaryHeader(digest.length);
                packer.writePayload(digest);
                break;
            default:
                packer.packString(key.value());
                break;
        }
    }

    /** The bytes of a hex digest. */
    private static byte[] hexBytes(String hex) {
        if (hex.length() % 2 != 0) {
            throw new IllegalArgumentException("a prompt-embeds digest is an even-length hex string, got " + hex);
        }
        byte[] bytes = new byte[hex.length() / 2];
        for (int i = 0; i < bytes.length; i++) {
            int hi = Character.digit(hex.charAt(2 * i), 16);
            int lo = Character.digit(hex.charAt(2 * i + 1), 16);
            if (hi < 0 || lo < 0) {
                throw new IllegalArgumentException("a prompt-embeds digest is a hex string, got " + hex);
            }
            bytes[i] = (byte) ((hi << 4) | lo);
        }
        return bytes;
    }

    private static void packHashes(MessageBufferPacker packer, byte[][] hashes, boolean intHashes)
            throws IOException {
        packer.packArrayHeader(hashes.length);
        for (byte[] hash : hashes) {
            packHash(packer, hash, intHashes);
        }
    }

    /** A block hash in vLLM's wire form: the unsigned leading 64 bits, or the raw digest bytes. */
    private static void packHash(MessageBufferPacker packer, byte[] hash, boolean intHashes) throws IOException {
        if (!intHashes) {
            packer.packBinaryHeader(hash.length);
            packer.writePayload(hash);
            return;
        }
        long value = KvCacheEvent.hashInt(hash);
        if (value >= 0) {
            packer.packLong(value);
        } else {
            packer.packBigInteger(BigInteger.valueOf(value).and(UNSIGNED_LONG_MASK));
        }
    }

    private static void packNullableString(MessageBufferPacker packer, String value) throws IOException {
        if (value == null) {
            packer.packNil();
        } else {
            packer.packString(value);
        }
    }
}
