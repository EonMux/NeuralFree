package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.Arrays;
import java.util.Objects;

/**
 * The whole-block chain hashes one prompt's {@link CacheRows} produced, kept so a submission that waits in the
 * queue does not hash its prompt again at every admission attempt ({@link PrefixClaims#reusable},
 * {@link PrefixClaims#diskReady}): block {@code i} (rows {@code [i * pageSize, (i + 1) * pageSize)}) keeps the hash it
 * took after the parent digest it was hashed after, and under the chain key a first block carries. A hash is reused
 * only for the same hasher, page size, parent digest and chain key, so a lookup returns exactly what hashing the
 * block again would. Thread-safe.
 */
final class BlockHashMemo {

    private BlockHasher hasher;
    private int pageSize;
    private byte[][] parents = new byte[0][];
    private byte[][] hashes = new byte[0][];
    private PrefixKey[] keys = new PrefixKey[0];

    /**
     * The memoized hash of whole block {@code block} hashed after {@code parent} (null for a chain's first block)
     * under {@code chainKey} by {@code hasher} at {@code pageSize}, or null when none is kept.
     */
    synchronized byte[] get(BlockHasher hasher, int pageSize, int block, byte[] parent, PrefixKey chainKey) {
        if (hasher != this.hasher || pageSize != this.pageSize || block >= hashes.length || hashes[block] == null) {
            return null;
        }
        if (!Arrays.equals(parents[block], parent) || !Objects.equals(keys[block], chainKey)) {
            return null;
        }
        return hashes[block];
    }

    /** Keeps {@code hash} as whole block {@code block}'s hash after {@code parent} under {@code chainKey}. */
    synchronized void put(BlockHasher hasher, int pageSize, int block, byte[] parent, PrefixKey chainKey,
                          byte[] hash) {
        if (hasher != this.hasher || pageSize != this.pageSize) {
            this.hasher = hasher;
            this.pageSize = pageSize;
            parents = new byte[0][];
            hashes = new byte[0][];
            keys = new PrefixKey[0];
        }
        if (block >= hashes.length) {
            int cap = Math.max(block + 1, hashes.length * 2);
            parents = Arrays.copyOf(parents, cap);
            hashes = Arrays.copyOf(hashes, cap);
            keys = Arrays.copyOf(keys, cap);
        }
        parents[block] = parent;
        hashes[block] = hash;
        keys[block] = chainKey;
    }
}
