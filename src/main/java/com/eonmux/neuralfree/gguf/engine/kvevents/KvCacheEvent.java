package com.eonmux.neuralfree.gguf.engine.kvevents;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One change of the prefix cache's block pool, in the shape of vLLM's {@code vllm/distributed/kv_events.py}:
 * {@link BlockStored} (blocks became reusable under their chain hashes), {@link BlockRemoved} (blocks left the
 * cache) and {@link AllBlocksCleared} (the whole cache was dropped). Block hashes are the block hasher's digests
 * ({@code BlockHasher}, vLLM's {@code hash_block_tokens}), the bytes vLLM's algorithms produce; the wire's integer
 * form reads the leading 64 bits big-endian, vLLM's {@code maybe_convert_block_hash}. Immutable.
 */
public abstract class KvCacheEvent {

    /** The medium of blocks kept in device memory (vLLM's {@code MEDIUM_GPU}). */
    public static final String MEDIUM_GPU = "GPU";
    /** The medium of blocks kept in host memory (vLLM's {@code MEDIUM_CPU}). */
    public static final String MEDIUM_CPU = "CPU";
    /** The medium of blocks kept as files in the disk tier (vLLM's {@code Medium.STORAGE}). */
    public static final String MEDIUM_STORAGE = "STORAGE";

    private KvCacheEvent() {
    }

    /**
     * Blocks stored in the cache, each extending the chain of the one before it: {@code blockHashes[0]} extends
     * {@code parentBlockHash} (null for a chain's first block), {@code tokenIds} holds the blocks' tokens in order,
     * {@code blockSize} tokens per block. A partial block — the trailing block of a finished sequence, shorter than
     * a page — is stored alone, with {@code blockSize} its token count (vLLM's partial-block entries).
     * {@code loraName} names the per-request adapters the blocks were computed under (vLLM's {@code lora_name}) and
     * {@code loraId} ids them (vLLM's deprecated {@code lora_id}), null for the base model. {@code extraKeys} holds
     * one entry per block, vLLM's untagged extra keys in the order of its {@code generate_block_hash_extra_keys}:
     * the adapters' content identity on a chain's first block computed under per-request adapters, the media items
     * the block covers, the cache salt of a chain's first block and the digest of the block's prompt-embedding rows
     * — or null for a block whose hash mixed in no extra key (later blocks inherit the adapters and the salt
     * through the chain). {@code kvCacheSpecKind} and {@code kvCacheSpecSlidingWindow} name the KV cache group the
     * blocks' pages live in (vLLM's per-group fields), null when its groups' specs disagree. Rows that hold no
     * token (media and prompt-embedding rows) carry {@code -1} in {@code tokenIds}, llama.cpp's
     * {@code LLAMA_TOKEN_NULL}.
     */
    public static final class BlockStored extends KvCacheEvent {
        private final byte[][] blockHashes;
        private final byte[] parentBlockHash;
        private final int[] tokenIds;
        private final int blockSize;
        private final String medium;
        private final Integer loraId;
        private final String loraName;
        private final String kvCacheSpecKind;
        private final Integer kvCacheSpecSlidingWindow;
        private final List<List<BlockExtraKey>> extraKeys;

        /**
         * @throws IllegalArgumentException when no block is stored, a hash is shorter than the 8 bytes the integer
         *                                  form reads, the tokens do not fill the blocks, or the extra keys are not
         *                                  one per block
         */
        public BlockStored(byte[][] blockHashes, byte[] parentBlockHash, int[] tokenIds, int blockSize, String medium,
                           Integer loraId, String loraName, String kvCacheSpecKind, Integer kvCacheSpecSlidingWindow,
                           List<List<BlockExtraKey>> extraKeys) {
            Objects.requireNonNull(blockHashes, "blockHashes");
            Objects.requireNonNull(tokenIds, "tokenIds");
            Objects.requireNonNull(extraKeys, "extraKeys");
            if (blockHashes.length == 0 || blockSize < 1 || tokenIds.length != blockHashes.length * blockSize
                    || extraKeys.size() != blockHashes.length) {
                throw new IllegalArgumentException(blockHashes.length + " stored blocks of " + blockSize
                        + " tokens need " + blockHashes.length * blockSize + " tokens and one extra key each, got "
                        + tokenIds.length + " tokens and " + extraKeys.size() + " keys");
            }
            for (byte[] hash : blockHashes) {
                checkHash("block_hashes", hash);
            }
            if (parentBlockHash != null) {
                checkHash("parent_block_hash", parentBlockHash);
            }
            this.blockHashes = copyHashes(blockHashes);
            this.parentBlockHash = parentBlockHash == null ? null : parentBlockHash.clone();
            this.tokenIds = tokenIds.clone();
            this.blockSize = blockSize;
            this.medium = medium;
            this.loraId = loraId;
            this.loraName = loraName;
            this.kvCacheSpecKind = kvCacheSpecKind;
            this.kvCacheSpecSlidingWindow = kvCacheSpecSlidingWindow;
            List<List<BlockExtraKey>> keys = new ArrayList<>(extraKeys.size());
            for (List<BlockExtraKey> block : extraKeys) {
                keys.add(block == null ? null : Collections.unmodifiableList(new ArrayList<>(block)));
            }
            this.extraKeys = Collections.unmodifiableList(keys);
        }

        /** The stored blocks' chain hashes, in chain order, the digests vLLM's algorithm produced. */
        public byte[][] blockHashes() { return copyHashes(blockHashes); }
        /** The hash of the block the first stored block extends, or null for a chain's first block. */
        public byte[] parentBlockHash() { return parentBlockHash == null ? null : parentBlockHash.clone(); }
        /** The stored blocks' tokens, in order. */
        public int[] tokenIds() { return tokenIds.clone(); }
        /** Tokens per stored block. */
        public int blockSize() { return blockSize; }
        /** {@link #MEDIUM_GPU}, {@link #MEDIUM_CPU} or {@link #MEDIUM_STORAGE}. */
        public String medium() { return medium; }
        /** The per-request adapters' id (vLLM's deprecated {@code lora_id}), or null for the base model. */
        public Integer loraId() { return loraId; }
        /** The per-request adapters the blocks were computed under (vLLM's {@code lora_name}), or null. */
        public String loraName() { return loraName; }
        /** The KV cache spec kind of the group the blocks live in (vLLM's {@code kv_cache_spec_kind}), or null. */
        public String kvCacheSpecKind() { return kvCacheSpecKind; }
        /** The group's sliding window (vLLM's {@code kv_cache_spec_sliding_window}), null for a full-attention group. */
        public Integer kvCacheSpecSlidingWindow() { return kvCacheSpecSlidingWindow; }
        /** The extra keys each block's hash mixed in, one entry per block, null for a block without any. */
        public List<List<BlockExtraKey>> extraKeys() { return extraKeys; }

        /** The stored blocks' hashes as vLLM's integer wire form: the leading 64 bits big-endian, unsigned. */
        public long[] blockHashInts() {
            long[] out = new long[blockHashes.length];
            for (int i = 0; i < blockHashes.length; i++) {
                out[i] = hashInt(blockHashes[i]);
            }
            return out;
        }

        /** {@link #parentBlockHash()} as vLLM's integer wire form, or null. */
        public Long parentBlockHashInt() {
            return parentBlockHash == null ? null : hashInt(parentBlockHash);
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof BlockStored)) {
                return false;
            }
            BlockStored e = (BlockStored) o;
            return hashesEqual(blockHashes, e.blockHashes) && Arrays.equals(parentBlockHash, e.parentBlockHash)
                    && Arrays.equals(tokenIds, e.tokenIds) && blockSize == e.blockSize
                    && Objects.equals(medium, e.medium) && Objects.equals(loraId, e.loraId)
                    && Objects.equals(loraName, e.loraName) && Objects.equals(kvCacheSpecKind, e.kvCacheSpecKind)
                    && Objects.equals(kvCacheSpecSlidingWindow, e.kvCacheSpecSlidingWindow)
                    && extraKeys.equals(e.extraKeys);
        }

        @Override
        public int hashCode() {
            return Objects.hash(Integer.valueOf(blockHashes.length), parentBlockHash == null ? 0
                            : Arrays.hashCode(parentBlockHash), Arrays.hashCode(tokenIds), blockSize, medium, loraId,
                    loraName, kvCacheSpecKind, kvCacheSpecSlidingWindow, extraKeys);
        }

        @Override
        public String toString() {
            return "BlockStored{block_hashes=" + hashesToString(blockHashes) + ", parent_block_hash="
                    + (parentBlockHash == null ? "null" : Long.toHexString(hashInt(parentBlockHash))) + ", token_ids="
                    + Arrays.toString(tokenIds) + ", block_size=" + blockSize + ", medium=" + medium
                    + (loraName == null ? "" : ", lora_name=" + loraName)
                    + (kvCacheSpecKind == null ? "" : ", kv_cache_spec_kind=" + kvCacheSpecKind)
                    + (kvCacheSpecSlidingWindow == null ? "" : ", kv_cache_spec_sliding_window="
                            + kvCacheSpecSlidingWindow) + ", extra_keys=" + extraKeys + "}";
        }
    }

    /** Blocks that left the cache: evicted for memory or by the token budget, or dropped with their memory. */
    public static final class BlockRemoved extends KvCacheEvent {
        private final byte[][] blockHashes;
        private final String medium;

        /** @throws IllegalArgumentException when no block is named or a hash is shorter than 8 bytes */
        public BlockRemoved(byte[][] blockHashes, String medium) {
            Objects.requireNonNull(blockHashes, "blockHashes");
            if (blockHashes.length == 0) {
                throw new IllegalArgumentException("a removal names at least one block");
            }
            for (byte[] hash : blockHashes) {
                checkHash("block_hashes", hash);
            }
            this.blockHashes = copyHashes(blockHashes);
            this.medium = medium;
        }

        /** The removed blocks' chain hashes, the digests vLLM's algorithm produced. */
        public byte[][] blockHashes() { return copyHashes(blockHashes); }
        /** {@link #MEDIUM_GPU}, {@link #MEDIUM_CPU} or {@link #MEDIUM_STORAGE}. */
        public String medium() { return medium; }
        /** The removed blocks' hashes as vLLM's integer wire form: the leading 64 bits big-endian, unsigned. */
        public long[] blockHashInts() {
            long[] out = new long[blockHashes.length];
            for (int i = 0; i < blockHashes.length; i++) {
                out[i] = hashInt(blockHashes[i]);
            }
            return out;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof BlockRemoved && hashesEqual(blockHashes, ((BlockRemoved) o).blockHashes)
                    && Objects.equals(medium, ((BlockRemoved) o).medium);
        }

        @Override
        public int hashCode() {
            return Objects.hash(Integer.valueOf(blockHashes.length), medium);
        }

        @Override
        public String toString() {
            return "BlockRemoved{block_hashes=" + hashesToString(blockHashes) + ", medium=" + medium + "}";
        }
    }

    /** Every cached block was dropped at once ({@code POST /reset_prefix_cache}, a memory reset). */
    public static final class AllBlocksCleared extends KvCacheEvent {
        /** The one instance: the event carries no data. */
        public static final AllBlocksCleared INSTANCE = new AllBlocksCleared();

        private AllBlocksCleared() {
        }

        @Override
        public String toString() {
            return "AllBlocksCleared{}";
        }
    }

    /** {@code hash} as vLLM's integer wire form: the leading 64 bits big-endian, unsigned. */
    public static long hashInt(byte[] hash) {
        long out = 0;
        for (int i = 0; i < 8; i++) {
            out = (out << 8) | (hash[i] & 0xffL);
        }
        return out;
    }

    private static byte[][] copyHashes(byte[][] hashes) {
        byte[][] out = new byte[hashes.length][];
        for (int i = 0; i < hashes.length; i++) {
            out[i] = hashes[i].clone();
        }
        return out;
    }

    private static boolean hashesEqual(byte[][] a, byte[][] b) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (!Arrays.equals(a[i], b[i])) {
                return false;
            }
        }
        return true;
    }

    private static void checkHash(String field, byte[] hash) {
        if (hash.length < 8) {
            throw new IllegalArgumentException(field + " needs at least the 8 bytes the integer form reads, got "
                    + hash.length);
        }
    }

    private static String hashesToString(byte[][] hashes) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < hashes.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(Long.toHexString(hashInt(hashes[i])));
        }
        return sb.append(']').toString();
    }
}
