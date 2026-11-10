package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

/**
 * A queued prompt's whole-block chain hashes are taken once ({@link BlockHashMemo}): every admission attempt
 * after the first ({@link PrefixCache#match}, the disk tier's prefetch) reuses the digests the first one took, and a
 * reused digest equals hashing the block again; a different chain key, parent digest, hasher or a partial block
 * hashes afresh.
 */
public class BlockHashMemoTest {

    private static final int PAGE = 4;
    private static final int CTX = 64;

    private static PrefixCache cache(BlockHasher hasher) {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16)
                .nCtx(CTX).ubatch(16).parallel(2).kvPageSize(PAGE).build();
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, CTX, 16)
                .group(MemoryPlan.KvGroup.of(0, CTX, false, KvWindow.FULL_CAUSAL, PAGE, CTX / PAGE))
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV)
                        .kv(KvLayout.of(CTX, false, 8, 8, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL), 0)
                        .build())
                .config(cfg)
                .build();
        PagedSequences paged = new PagedSequences(plan, 2);
        return new PrefixCache(paged, 0, "GPU", false, -1, hasher, null, null, null,
                new BlockLayout(paged, List.of()));
    }

    private static CacheRows prompt() {
        int[] t = new int[10 * PAGE + 3];
        for (int i = 0; i < t.length; i++) {
            t[i] = 11 + (i * 7) % 50;
        }
        return CacheRows.ofTokens(t);
    }

    @Test
    public void aRepeatedLookupReusesTheChainsDigests() {
        BlockHasher hasher = BlockHasher.of(PrefixHashAlgo.SHA256, BlockHasher.DEFAULT_NONE_HASH_SEED);
        PrefixCache cache = cache(hasher);
        CacheRows rows = prompt();
        PrefixKey key = PrefixKey.NONE;
        // the disk tier's walk of the whole chain, as an admission attempt takes it
        int blocks = rows.length() / PAGE;
        byte[][] walked = new byte[blocks][];
        byte[] parent = null;
        for (int b = 0; b < blocks; b++) {
            walked[b] = PrefixCache.memoHash(hasher, PAGE, parent, rows, b * PAGE, PAGE, b == 0 ? key : null);
            parent = walked[b];
        }
        parent = null;
        for (int b = 0; b < blocks; b++) {
            PrefixKey chainKey = b == 0 ? key : null;
            assertArrayEquals("block " + b + " keeps the digest hashing it again takes",
                    cache.hash(parent, rows, b * PAGE, PAGE, chainKey), walked[b]);
            assertSame("the next attempt takes block " + b + "'s kept digest", walked[b],
                    PrefixCache.memoHash(hasher, PAGE, parent, rows, b * PAGE, PAGE, chainKey));
            parent = walked[b];
        }
        cache.match(rows, key, rows.length() - 1);
        assertSame("the memory's lookup takes the kept digest", walked[0], cache.hashOf(null, rows, 0, PAGE, key));
    }

    @Test
    public void anotherKeyParentHasherOrAPartialBlockHashesAfresh() {
        BlockHasher hasher = BlockHasher.of(PrefixHashAlgo.SHA256, BlockHasher.DEFAULT_NONE_HASH_SEED);
        PrefixCache cache = cache(hasher);
        CacheRows rows = prompt();
        byte[] first = cache.hashOf(null, rows, 0, PAGE, PrefixKey.NONE);
        PrefixKey salted = PrefixKey.restored(null, null, "salt");
        byte[] saltedHash = cache.hashOf(null, rows, 0, PAGE, salted);
        assertNotSame(first, saltedHash);
        assertArrayEquals(cache.hash(null, rows, 0, PAGE, salted), saltedHash);
        assertNull("the salted digest replaced the unsalted one",
                rows.hashMemo().get(hasher, PAGE, 0, null, PrefixKey.NONE));

        byte[] second = cache.hashOf(null, rows, PAGE, PAGE, null);
        assertArrayEquals(cache.hash(null, rows, PAGE, PAGE, null), second);
        assertNull("another parent digest misses",
                rows.hashMemo().get(hasher, PAGE, 1, first, null));

        BlockHasher other = BlockHasher.of(PrefixHashAlgo.XXHASH, BlockHasher.DEFAULT_NONE_HASH_SEED);
        assertNull("another hasher misses", rows.hashMemo().get(other, PAGE, 1, null, null));
        assertNull("another page size misses", rows.hashMemo().get(hasher, 2 * PAGE, 1, null, null));

        byte[] partial = cache.hashOf(null, rows, 0, PAGE - 1, PrefixKey.NONE);
        assertArrayEquals(cache.hash(null, rows, 0, PAGE - 1, PrefixKey.NONE), partial);
        assertSame("a partial block leaves the whole block's digest", saltedHash,
                rows.hashMemo().get(hasher, PAGE, 0, null, salted));
    }
}
