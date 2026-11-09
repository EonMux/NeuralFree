package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The host-RAM and disk tiers of the prefix cache's block pool (vLLM's tiering filesystem tier beside its
 * {@code block_pool}): blocks the pool evicts for room move to the host tier ({@link HostBlockCache}) through the
 * {@link BlockLayout}, the host tier's own drops move on to the disk tier ({@link DiskBlockCache}, one file per
 * block), and a later prompt walks them back into fresh pages ({@link #promote}). The tiers read the pool's chain
 * lookups through its package-private hooks and queue their KV cache events on it, so a block keeps one identity
 * (its chain hash) across the memory, the host and the disk.
 */
final class BlockTiers {

    /** The pool whose evicted blocks these tiers offload and whose chains they extend. */
    private final PrefixCache pool;
    /** The host-RAM tier evicted blocks move to, or null without one. */
    private final HostBlockCache host;
    /** The disk tier the host-RAM tier's drops move to, or null without one. */
    private final DiskBlockCache disk;
    /** How blocks move between the memory and the host; null when no block leaves the memory. */
    private final BlockLayout layout;
    /** The block hasher the chain hashes and the tier keys come from. */
    private final BlockHasher hasher;
    private final int pageSize;

    BlockTiers(PrefixCache pool, HostBlockCache host, DiskBlockCache disk, BlockLayout layout, BlockHasher hasher,
               int pageSize) {
        this.pool = pool;
        this.host = host;
        this.disk = disk;
        this.layout = layout;
        this.hasher = hasher;
        this.pageSize = pageSize;
    }

    /**
     * Moves the rows of {@code b}, about to be evicted for room, to the host-RAM tier, queuing its
     * {@code BlockStored} on the host medium and a {@code BlockRemoved} for every host block the budget dropped. A
     * block whose chain holds media or prompt-embedding rows stays out: the tier carries token chains only. A block
     * holding a context checkpoint stays out too: its checkpoint's window pages and recurrent state do not move
     * with its rows, so the host copy alone would not restore. The drops the host budget made, and the block itself
     * when even the host keeps none of it, move on to the disk tier when there is one; the disk writes run behind
     * the engine and publish their {@code BlockStored} once durable ({@link #pumpDisk}).
     */
    void offload(PrefixCache.Block b) {
        if (host == null || !b.checkpoints.isEmpty()) {
            return;
        }
        int[] chain = chainTokens(b);
        for (int t : chain) {
            if (t == CacheRows.NO_TOKEN) {
                return;
            }
        }
        BlockImage image = BlockImage.read(b, b.rows.length(), b.full, layout);
        PrefixKey key = PrefixCache.chainKey(b);
        HostBlockCache.Entry entry = new HostBlockCache.Entry(b.hash, parentHashOf(b.parent), key, chain, image);
        List<HostBlockCache.Entry> dropped = host.put(entry);
        spill(dropped);
        if (!pool.eventsOn()) {
            return;
        }
        PrefixCache.Spec spec = pool.specFor(b.pages);
        if (!dropped.contains(entry)) {
            pool.queueEvent(new KvCacheEvent.BlockStored(new byte[][] {b.hash},
                    b.parent == null ? null : b.parent.hash, image.tokens(), image.rows(), KvCacheEvent.MEDIUM_CPU,
                    key == null ? null : key.loraId(), key == null ? null : key.loraName(),
                    spec == null ? null : spec.kind, spec == null ? null : spec.window,
                    Collections.singletonList(b.extraKeys)));
        }
        for (HostBlockCache.Entry d : dropped) {
            if (d != entry) {
                pool.queueEvent(new KvCacheEvent.BlockRemoved(new byte[][] {d.hash}, KvCacheEvent.MEDIUM_CPU));
            }
        }
    }

    /**
     * Hands the host-tier drops {@code dropped} to the disk tier's background writer. A full queue, a write failure
     * or a block larger than the disk budget keeps no copy; nothing here waits for the disk.
     */
    private void spill(List<HostBlockCache.Entry> dropped) {
        if (disk == null || dropped.isEmpty()) {
            return;
        }
        for (HostBlockCache.Entry e : dropped) {
            disk.put(e);
        }
    }

    /**
     * Drains the disk tier's completed work onto the event queue: a block whose file became durable queues its
     * {@code BlockStored} on the storage medium, and a file the budget dropped queues a {@code BlockRemoved}. A
     * storage store therefore always follows its file's durability, whatever the device and host events queued in
     * between.
     */
    void pumpDisk() {
        if (disk == null) {
            return;
        }
        for (DiskBlockCache.Event e : disk.pump()) {
            if (e.stored != null) {
                storedEvents(Collections.singletonList(e.stored));
            } else if (pool.eventsOn()) {
                pool.queueEvent(new KvCacheEvent.BlockRemoved(new byte[][] {e.removed},
                        KvCacheEvent.MEDIUM_STORAGE));
            }
        }
    }

    /**
     * Queues a disk-tier {@code BlockStored} for every entry of {@code stored}, with the fields the device and host
     * media publish. The entry's pages are gone, so the KV cache spec is the one over every group
     * ({@code specFor} on no pages).
     */
    private void storedEvents(List<HostBlockCache.Entry> stored) {
        if (!pool.eventsOn()) {
            return;
        }
        for (HostBlockCache.Entry e : stored) {
            boolean root = Arrays.equals(e.parentHash, hasher.noneHash());
            PrefixCache.Spec spec = pool.specFor(new int[0]);
            pool.queueEvent(new KvCacheEvent.BlockStored(new byte[][] {e.hash},
                    root ? null : e.parentHash, e.image.tokens(), e.image.rows(),
                    KvCacheEvent.MEDIUM_STORAGE, e.key.loraId(), e.key.loraName(),
                    spec == null ? null : spec.kind, spec == null ? null : spec.window,
                    Collections.singletonList(CacheRows.ofTokens(e.image.tokens())
                            .extraKeys(0, e.image.rows(), root ? e.key : null))));
        }
    }

    /**
     * Starts the background reads a later {@link #promote} of the prefix {@code rows[0, maxTokens)} under
     * {@code key} would need, and answers whether every block the disk tier can contribute is already in memory.
     * The scheduler delays the request while this answers false, so the engine's thread never waits on the disk;
     * a promotion that still runs early takes what is loaded and recomputes the rest. True without a disk tier.
     */
    boolean prefetch(CacheRows rows, PrefixKey key, int maxTokens) {
        if (disk == null) {
            return true;
        }
        pumpDisk();
        int limit = Math.min(maxTokens, rows.length());
        if (limit <= 0) {
            return true;
        }
        int[] want = rows.tokens(0, limit);
        return disk.prefetchReady(want, key, limit, pageSize, servedBlocks(rows, key, want, limit));
    }

    /**
     * The chain hashes of {@code want[0, limit)} the memory or the host tier already serves, which the disk tier
     * skips when it plans the reads a promotion needs: a block the host holds is not read off the disk again.
     */
    private Set<HostBlockCache.DigestKey> servedBlocks(CacheRows rows, PrefixKey key, int[] want, int limit) {
        Set<HostBlockCache.DigestKey> served = new HashSet<>();
        PrefixCache.Block parent = null;
        byte[] parentHash = hasher.noneHash();
        boolean device = true;
        int pos = 0;
        while (pos + pageSize <= limit) {
            byte[] h;
            boolean onTier;
            if (device) {
                PrefixCache.Block b = pool.find(parent, rows, pos, pageSize, key);
                if (b != null && b.full) {
                    parent = b;
                    h = b.hash;
                    onTier = true;
                } else {
                    device = false;
                    h = pool.hashOf(parent, rows, pos, pageSize, parent == null ? key : null);
                    onTier = host != null && host.whole(parentHash, key, want, pos, pageSize) != null;
                }
            } else {
                h = PrefixCache.chainHash(hasher, parentHash, rows, pos, pageSize, null);
                onTier = host != null && host.whole(parentHash, key, want, pos, pageSize) != null;
            }
            if (onTier) {
                served.add(new HostBlockCache.DigestKey(h));
            }
            parentHash = h;
            pos += pageSize;
        }
        return served;
    }

    /**
     * Brings back the host-RAM blocks that extend the cached prefix of {@code rows} under {@code key}, at most
     * {@code maxTokens} rows deep, so the next {@link PrefixCache#match} finds them in the memory: whole blocks
     * along the chain where the pool holds none, then the boundary block when the host or the disk shares more of
     * its rows than any cached block does. A disk block loads into the host tier first and leaves the disk. Each
     * takes fresh pages (free cached blocks evicted for room, the walked chain spared) and leaves its tier;
     * promotion stops where no page can be freed. Disk blocks come from the images a prefetch already loaded; a
     * block still on the disk misses here and simply recomputes. No-op without a tier.
     *
     * @return the blocks brought back
     */
    int promote(CacheRows rows, PrefixKey key, int maxTokens) {
        pumpDisk();
        if (host == null || (host.size() == 0 && (disk == null || disk.isEmpty()))) {
            return 0;
        }
        pool.validate();
        int limit = Math.min(maxTokens, rows.length());
        if (limit <= 0) {
            return 0;
        }
        int[] want = rows.tokens(0, limit);
        Set<PrefixCache.Block> keep = Collections.newSetFromMap(new IdentityHashMap<>());
        PrefixCache.Block parent = null;
        int pos = 0;
        int promoted = 0;
        while (pos + pageSize <= limit) {
            PrefixCache.Block b = pool.find(parent, rows, pos, pageSize, key);
            if (b == null || !b.full) {
                HostBlockCache.Entry e = host.whole(parentHashOf(parent), key, want, pos, pageSize);
                if (e == null) {
                    e = disk == null ? null : disk.wholeLoaded(parentHashOf(parent), key, want, pos, pageSize);
                    if (e != null) {
                        load(e);
                    }
                }
                if (e == null || !pool.makeRoom(keep)) {
                    return promoted;
                }
                b = bringBack(parent, e, key, keep);
                promoted++;
            }
            keep.add(b);
            parent = b;
            pos += pageSize;
        }
        if (pos < limit) {
            HostBlockCache.Hit hit = bestBoundary(parentHashOf(parent), key, want, pos, limit);
            if (hit != null && hit.rows > deviceShare(parent, rows, pos, limit, key) && pool.makeRoom(keep)) {
                load(hit.entry);
                bringBack(parent, hit.entry, key, keep);
                promoted++;
            }
        }
        return promoted;
    }

    /** The boundary block of the two tiers sharing the most of {@code want[pos, limit)}, the host winning ties. */
    private HostBlockCache.Hit bestBoundary(byte[] parentHash, PrefixKey key, int[] want, int pos, int limit) {
        HostBlockCache.Hit hit = host.best(parentHash, key, want, pos, limit);
        HostBlockCache.Hit fromDisk = disk == null ? null : disk.bestLoaded(parentHash, key, want, pos, limit);
        if (fromDisk == null) {
            return hit;
        }
        return hit != null && hit.rows >= fromDisk.rows ? hit : fromDisk;
    }

    /**
     * Moves the disk block {@code e} into the host tier, publishing the move when the block was durable. A block
     * whose file is still being written leaves the write queue: it never reaches the storage medium, so no storage
     * event is owed for it.
     */
    private void load(HostBlockCache.Entry e) {
        if (disk == null || !disk.holds(e.hash)) {
            return;
        }
        boolean durable = disk.durable(e.hash);
        disk.removeAsync(e.hash);
        if (pool.eventsOn() && durable) {
            pool.queueEvent(new KvCacheEvent.BlockRemoved(new byte[][] {e.hash}, KvCacheEvent.MEDIUM_STORAGE));
        }
        spill(host.put(e));
    }

    /** Installs a host entry after {@code parent}; the entry leaves the tier. */
    private PrefixCache.Block bringBack(PrefixCache.Block parent, HostBlockCache.Entry e, PrefixKey key,
                                        Set<PrefixCache.Block> keep) {
        host.remove(e);
        return pool.install(parent, e.image, key, layout, keep);
    }

    /** The most rows of {@code rows[pos, limit)} a cached block after {@code parent} shares. */
    private int deviceShare(PrefixCache.Block parent, CacheRows rows, int pos, int limit, PrefixKey key) {
        int best = 0;
        for (PrefixCache.Block c : pool.siblings(parent, key)) {
            best = Math.max(best, CacheRows.commonLength(c.rows, 0, rows, pos, Math.min(c.rows.length(),
                    limit - pos)));
        }
        return best;
    }

    /**
     * Host copies of the cached prefix of the token rows {@code rows} under {@code key}, at most {@code maxTokens}
     * rows: whole blocks along the chain from the memory, or from the host-RAM or disk tier where the memory holds
     * none, then the shared rows of the boundary block (memory, host or disk). Reads only; nothing moves.
     *
     * @throws UnsupportedOperationException when a block holds media or prompt-embedding rows
     */
    List<BlockImage> images(CacheRows rows, PrefixKey key, int maxTokens) {
        pool.validate();
        List<BlockImage> out = new ArrayList<>();
        int limit = Math.min(maxTokens, rows.length());
        if (limit <= 0) {
            return out;
        }
        int[] want = rows.tokens(0, limit);
        PrefixCache.Block parent = null;
        boolean inMemory = true;
        byte[] parentHash = hasher.noneHash();
        int pos = 0;
        while (pos + pageSize <= limit) {
            PrefixCache.Block b = inMemory ? pool.find(parent, rows, pos, pageSize, key) : null;
            if (b != null && b.full) {
                out.add(BlockImage.read(b, pageSize, true, layout));
                parent = b;
                parentHash = b.hash;
            } else {
                HostBlockCache.Entry e = host == null ? null : host.whole(parentHash, key, want, pos, pageSize);
                if (e == null && disk != null) {
                    e = disk.whole(parentHash, key, want, pos, pageSize);
                }
                if (e == null) {
                    break;
                }
                out.add(e.image);
                inMemory = false;
                parentHash = e.hash;
            }
            pos += pageSize;
        }
        if (pos < limit) {
            boundary(out, inMemory ? parent : null, inMemory, parentHash, rows, want, pos, limit, key);
        }
        return out;
    }

    /** Adds the boundary rows {@link #images} ends with: the block, in memory, on the host or on the disk, sharing
     *  the most. */
    private void boundary(List<BlockImage> out, PrefixCache.Block parent, boolean inMemory, byte[] parentHash,
                          CacheRows rows, int[] want, int pos, int limit, PrefixKey key) {
        PrefixCache.Block best = null;
        int bestLen = 0;
        if (inMemory) {
            for (PrefixCache.Block c : pool.siblings(parent, key)) {
                int len = CacheRows.commonLength(c.rows, 0, rows, pos, Math.min(c.rows.length(), limit - pos));
                if (len > bestLen || (len == bestLen && len > 0 && c.lastUse > best.lastUse)) {
                    best = c;
                    bestLen = len;
                }
            }
        }
        HostBlockCache.Hit hit = host == null ? null : host.best(parentHash, key, want, pos, limit);
        HostBlockCache.Hit fromDisk = disk == null ? null : disk.best(parentHash, key, want, pos, limit);
        if (fromDisk != null && (hit == null || fromDisk.rows > hit.rows)) {
            hit = fromDisk;
        }
        if (hit != null && hit.rows > bestLen) {
            BlockImage image = hit.entry.image;
            out.add(hit.rows == image.rows() ? image : image.head(hit.rows, layout));
        } else if (best != null) {
            out.add(BlockImage.read(best, bestLen, bestLen == pageSize, layout));
        }
    }

    /**
     * Drops the host-RAM and disk blocks whose whole chain lies within the first {@code count} rows of
     * {@code rows} under {@code key}, except those {@code spare} answers true for (a prefix another history still
     * walks).
     */
    void forgetHost(CacheRows rows, PrefixKey key, int count, Predicate<HostBlockCache.Entry> spare) {
        if (host == null) {
            return;
        }
        int[] want = rows.tokens(0, Math.min(count, rows.length()));
        for (HostBlockCache.Entry e : host.within(want, key)) {
            if (!spare.test(e)) {
                host.remove(e);
                if (pool.eventsOn()) {
                    pool.queueEvent(new KvCacheEvent.BlockRemoved(new byte[][] {e.hash}, KvCacheEvent.MEDIUM_CPU));
                }
            }
        }
        if (disk == null) {
            return;
        }
        for (HostBlockCache.Entry e : disk.within(want, key)) {
            if (!spare.test(e)) {
                disk.remove(e.hash);
                if (pool.eventsOn()) {
                    pool.queueEvent(new KvCacheEvent.BlockRemoved(new byte[][] {e.hash},
                            KvCacheEvent.MEDIUM_STORAGE));
                }
            }
        }
    }

    /** The tokens from {@code b}'s chain root through its own rows. */
    private static int[] chainTokens(PrefixCache.Block b) {
        List<PrefixCache.Block> path = new ArrayList<>();
        int n = 0;
        for (PrefixCache.Block x = b; x != null; x = x.parent) {
            path.add(x);
            n += x.rows.length();
        }
        int[] out = new int[n];
        int at = 0;
        for (int i = path.size() - 1; i >= 0; i--) {
            int[] t = path.get(i).rows.tokens(0, path.get(i).rows.length());
            System.arraycopy(t, 0, out, at, t.length);
            at += t.length;
        }
        return out;
    }

    /** The digest a child of {@code parent} is keyed under: the hasher's NONE_HASH for a chain's first block. */
    private byte[] parentHashOf(PrefixCache.Block parent) {
        return parent == null ? hasher.noneHash() : parent.hash;
    }

    /** Drops every host-RAM and disk block; the pool's clear takes the tiers with it. */
    void clear() {
        if (host != null) {
            host.clear();
        }
        if (disk != null) {
            disk.clear();
        }
    }

    /**
     * Waits at most {@code millis} for the disk tier's admitted writes to become durable and queues their events;
     * true when every write finished. A drain uses it so a finished engine still leaves its spills on the disk.
     */
    boolean awaitDisk(long millis) {
        if (disk == null) {
            return true;
        }
        boolean idle = disk.awaitWrites(millis);
        pumpDisk();
        return idle;
    }

    /**
     * Stops the disk tier's background workers, letting the writes it accepted finish for a bounded time and
     * abandoning the rest, then drains what completed. The engine's states stay as they are.
     */
    void close() {
        if (disk == null) {
            return;
        }
        disk.close();
        pumpDisk();
    }
}
