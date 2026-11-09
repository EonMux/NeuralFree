package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.paged.PageWindow;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The hash-keyed block pool of the batch engine's prefix cache (vLLM's {@code block_pool.py} over the block hash
 * chain of {@code kv_cache_utils.py}): a block is one page per KV group holding {@code P} consecutive rows of a
 * sequence, keyed by the chain hash of its parent's hash and its rows ({@link CacheRows}) — a token row by its
 * token, a media row by its item's identity (the client's uuid or a content digest) and its index in the item, a
 * prompt-embedding row by its own digest — and the first block of a chain mixes in the request's extra keys
 * ({@link PrefixKey}: its per-request adapters' content identity and its {@code cache_salt}), which every later block
 * inherits through its parent's hash. The chain hash is vLLM's {@code hash_block_tokens} over the configured
 * algorithm ({@link BlockHasher}), so the hashes the events publish are the ones a router recomputes from the token
 * ids. So requests under different adapters or salts never share a block, and different
 * media never do. The pool holds its own page reference on every cached block ({@link PagedSequences#holdPages}), so a
 * block outlives the sequence that computed it and any later prompt whose prefix walks the same chain adopts it,
 * whichever history it came from.
 *
 * <p>Running sequences store their blocks as they fill and commit ({@link #commit}, vLLM's
 * {@code cache_full_blocks}); a retiring or preempted sequence also stores its trailing partial block
 * ({@link #commitPartial}), which never extends further. A lookup ({@link #match}) walks the chain over whole
 * blocks, then shares the longest token prefix of one cached block at the boundary: that page joins the adopter up
 * to the shared rows and its first write copies the rows it owns (copy-on-write). A match never ends inside a
 * media item ({@link CacheRows#cut}): the item prefills whole. Every match is verified against the stored rows'
 * tokens and keys, the parent and the extra keys, so a hash collision can only miss, never share foreign rows.
 *
 * <p>Blocks no sequence uses (only the pool's reference remains) are the free cached blocks: the least recently
 * used free leaf goes first when memory runs short ({@link #evictOne}), a leaf so no cached block outlives its
 * parent, and a token budget bounds the tokens free blocks hold together ({@link #trimToBudget}). Every store,
 * eviction and clear queues a KV cache event ({@link #takeEvents}) when events are on, and a request that asks for
 * the full report mode also queues one for each cached block it reuses ({@link #reportReused}). A memory clear
 * ({@link PagedSequences#clears()}) drops the whole pool before its next use. Not thread-safe: the engine thread
 * owns it.
 *
 * <p>Over a memory with recurrent state or sliding-window groups the pool runs with {@link ContextCheckpoints}: a
 * block then holds the pages of the full-attention groups only, and a prefix is restorable only where a
 * {@link ContextCheckpoint} hangs on the block chain at exactly its rows ({@link #checkpoint}). A lookup then
 * answers the longest such checkpoint within the shared rows (vLLM's hybrid coordinator settles on the longest hit
 * every group can serve), adoption writes the checkpoint's recurrent state and window pages into the adopter, and a
 * block takes its checkpoints along when it leaves the pool. Eviction prefers free leaves without a checkpoint: no
 * prompt can restore at them. At most {@link ContextCheckpoints#limit()} checkpoints stay, the least recently used
 * dropped first.
 */
public final class PrefixCache {

    /**
     * One cached block: its chain hash (the block hasher's digest, the bytes vLLM's algorithm produced), the 64-bit
     * map key the pool reads from its leading bytes, parent, rows, the extra keys of a chain's first block, its
     * pages and the extra keys its events publish.
     */
    static final class Block {
        final byte[] hash;
        /** The map key: the leading 64 bits of {@code hash} big-endian, unsigned (the integer wire form). */
        final long mapKey;
        final Block parent;
        final CacheRows rows;
        final boolean full;
        /** The extra keys of a chain's first block; null for later blocks (they inherit them through the chain). */
        final PrefixKey key;
        final int[] pages;
        /** The first row of the block in its chain (its parent's blocks are whole). */
        final int start;
        /** The context checkpoints hanging on the block (their rows end inside it or at its end). */
        final List<ContextCheckpoint> checkpoints = new ArrayList<>(0);
        /** The block's extra keys as {@code BlockStored} publishes them; null when the block mixes in none. */
        final List<BlockExtraKey> extraKeys;
        final List<Block> children = new ArrayList<>(2);
        Block nextSameHash;
        boolean cached = true;
        long lastUse;

        Block(byte[] hash, Block parent, CacheRows rows, boolean full, PrefixKey key, int[] pages,
              List<BlockExtraKey> extraKeys) {
            this.hash = hash;
            this.mapKey = KvCacheEvent.hashInt(hash);
            this.parent = parent;
            this.rows = rows;
            this.full = full;
            this.key = key;
            this.pages = pages;
            this.extraKeys = extraKeys;
            this.start = parent == null ? 0 : parent.start + parent.rows.length();
        }
    }

    /**
     * The blocks a sequence's committed rows extend: the last whole block stored or matched for it, how many whole
     * blocks lead to it, and whether storing stopped (its rows no longer map onto the chain). Owned by one slot.
     */
    static final class BlockChain {
        Block tail;
        int blocks;
        boolean stopped;
    }

    /**
     * A lookup's answer: the whole blocks the prompt's prefix walks, then the cached block the shared prefix ends
     * inside and how many of its rows the prompt shares (0 with no such block).
     */
    static final class Match {
        static final Match NONE = new Match(Collections.emptyList(), null, 0, 0, null);

        final List<Block> blocks;
        final Block partial;
        final int partialRows;
        /** The checkpoint the match restores at its end, or null for a pool without checkpoints. */
        final ContextCheckpoint checkpoint;
        private final int pageSize;

        Match(List<Block> blocks, Block partial, int partialRows, int pageSize, ContextCheckpoint checkpoint) {
            this.blocks = blocks;
            this.partial = partial;
            this.partialRows = partialRows;
            this.pageSize = pageSize;
            this.checkpoint = checkpoint;
        }

        /** Prompt tokens the match lets an adopter skip. */
        int tokens() {
            return blocks.size() * pageSize + partialRows;
        }

        /** Every matched block, the boundary block last: the blocks eviction must spare while the match stands. */
        Set<Block> held() {
            Set<Block> held = Collections.newSetFromMap(new IdentityHashMap<>());
            held.addAll(blocks);
            if (partial != null) {
                held.add(partial);
            }
            return held;
        }
    }

    private final PagedSequences paged;
    private final int pageSize;
    private final int budget;
    private final String medium;
    private final boolean events;
    /** The block hasher (vLLM's hash function and NONE_HASH seed) the chain hashes come from. */
    private final BlockHasher hasher;
    /** The KV cache spec kind of every KV group, the cache-spec metadata the events publish. */
    private final String[] groupKinds;
    /** The sliding window of every KV group, 0 for a full-attention group. */
    private final int[] groupWindows;
    /** The id an adopted embedding row records in its sequence's token record. */
    private final int embeddingToken;
    private final Map<Long, Block> byHash = new HashMap<>();
    /** The first blocks of every chain (they have no parent to list them). */
    private final List<Block> roots = new ArrayList<>();
    /** Every cached block, least recently used first. */
    private final LinkedHashSet<Block> lru = new LinkedHashSet<>();
    private final List<KvCacheEvent> pending = new ArrayList<>();
    /** The context checkpoints of a memory that needs them, or null. */
    private final ContextCheckpoints checkpoints;
    /** Every checkpoint hanging on a cached block, least recently used first. */
    private final LinkedHashSet<ContextCheckpoint> checkpointLru = new LinkedHashSet<>();
    /** The host-RAM and disk tiers the pool's evicted blocks move through ({@link BlockTiers}). */
    private final BlockTiers tiers;
    /** Every checkpoint in creation order, oldest first (llama.cpp's per-slot checkpoint list). */
    private final List<ContextCheckpoint> checkpointAge = new ArrayList<>();
    private int clears;
    private long clock;

    /**
     * A pool over {@code paged} whose free blocks hold at most {@code budget} tokens together ({@code <= 0} leaves
     * the pools the only bound); {@code medium} names where the blocks live in events, which are queued only when
     * {@code events} is set; an adopted embedding row records {@code embeddingToken} in its sequence's token
     * record, the id the model's forward records for one ({@code -1} for none); {@code hasher} computes the chain
     * hashes (vLLM's algorithm and NONE_HASH seed). Prefixes restore only at the context checkpoints of
     * {@code checkpoints} when it is non-null, at any row with null. Blocks evicted for room move to {@code host}
     * (the host-RAM tier) through {@code layout}; a null {@code host} drops them, and the host tier's own drops
     * move on to {@code disk} (the disk tier) through the same layout.
     *
     * @throws IllegalArgumentException for a negative budget, or a host tier without a layout
     */
    PrefixCache(PagedSequences paged, int budget, String medium, boolean events, int embeddingToken,
                BlockHasher hasher, ContextCheckpoints checkpoints, HostBlockCache host, DiskBlockCache disk,
                BlockLayout layout) {
        this.paged = Objects.requireNonNull(paged, "paged");
        this.hasher = Objects.requireNonNull(hasher, "hasher");
        this.checkpoints = checkpoints;
        if (budget < 0) {
            throw new IllegalArgumentException("prefix cache budget must be >= 0 (0 = unbounded), got " + budget);
        }
        if (host != null && layout == null) {
            throw new IllegalArgumentException("a host-RAM tier needs the layout its blocks move through");
        }
        if (disk != null && (host == null || layout == null)) {
            throw new IllegalArgumentException("a disk tier runs under a host-RAM tier over one layout");
        }
        this.tiers = new BlockTiers(this, host, disk, layout, hasher, paged.pageSize());
        this.pageSize = paged.pageSize();
        this.budget = budget;
        this.medium = medium;
        this.events = events;
        this.embeddingToken = embeddingToken;
        int groups = paged.groups();
        this.groupKinds = new String[groups];
        this.groupWindows = new int[groups];
        for (int g = 0; g < groups; g++) {
            groupKinds[g] = specKind(paged, g);
            groupWindows[g] = paged.trims(g) ? paged.windowWidth(g) : 0;
        }
        this.clears = paged.clears();
    }

    /** Tokens per whole block (the page size). */
    int pageSize() {
        return pageSize;
    }

    /** Cached blocks, whole and partial. */
    int size() {
        validate();
        return lru.size();
    }

    /** Context checkpoints hanging on cached blocks. */
    int checkpointCount() {
        validate();
        return checkpointLru.size();
    }

    /** Whether prefixes restore only at context checkpoints. */
    boolean checkpointed() {
        return checkpoints != null;
    }

    // ------------------------------------------------------------------ hashing

    /**
     * The hash of the block holding rows {@code [from, from + len)} of {@code rows} after the block hashed
     * {@code parentBytes}: vLLM's {@code hash_block_tokens} over the parent digest (the hasher's {@code NONE_HASH}
     * for a chain's first block, null here), the rows' token ids and the tagged extra keys — a chain's first
     * block's {@code chainKey} (its adapters and cache salt), the media items and prompt-embedding rows the block
     * covers.
     */
    byte[] hash(byte[] parentBytes, CacheRows rows, int from, int len, PrefixKey chainKey) {
        return chainHash(hasher, parentBytes, rows, from, len, chainKey);
    }

    /** {@link #hash} over any hasher: the form the disk tier verifies a block file's name with. */
    static byte[] chainHash(BlockHasher hasher, byte[] parentBytes, CacheRows rows, int from, int len,
                            PrefixKey chainKey) {
        List<BlockExtraKey> keys = rows.extraKeys(from, from + len, chainKey);
        return hasher.hash(parentBytes, rows.tokens(from, from + len), keys);
    }

    /** The hash of the block after the cached block {@code parent} hashed, its chain key only on a first block. */
    byte[] hashOf(Block parent, CacheRows rows, int from, int len, PrefixKey chainKey) {
        return hash(parent == null ? null : parent.hash, rows, from, len, chainKey);
    }

    /**
     * The KV cache spec kind of group {@code g} (vLLM's {@code get_kv_cache_spec_kind}, the values its
     * {@code KVCacheSpecKind} names): a group whose layers share one sliding-window rule by it
     * ({@code sliding_window}, {@code chunked_local_attention}), a group of MLA latent layers that keeps every
     * position {@code mla_attention}, anything else {@code full_attention}; null when the group holds no KV layer.
     */
    static String specKind(PagedSequences paged, int g) {
        boolean mla = false;
        int kvLayers = 0;
        SwaType windowType = null;
        boolean mixedWindows = false;
        for (LayerMemoryPlan layer : paged.plan().layers()) {
            if (layer.kvGroup() != g || layer.kv() == null) {
                continue;
            }
            kvLayers++;
            mla |= layer.kv().vAliasesK();
            SwaType type = layer.kv().window().type();
            if (windowType == null) {
                windowType = type;
            } else if (windowType != type) {
                mixedWindows = true;
            }
        }
        if (kvLayers == 0) {
            return null;
        }
        if (mixedWindows) {
            // layers of differing window rules keep every position: vLLM's mixed-spec group reads as full attention
            return "full_attention";
        }
        switch (windowType) {
            case STANDARD:
                return mla ? "sliding_window_mla" : "sliding_window";
            case CHUNKED:
            case CHUNKED_LOOKBACK:
                return "chunked_local_attention";
            case SYMMETRIC:
                return "sliding_window";
            default:
                return mla ? "mla_attention" : "full_attention";
        }
    }

    /**
     * The KV cache spec metadata of the groups the block's {@code pages} hold pages for (all groups when none
     * holds any, a cache whose every group trims): the kind when those groups agree, else null, and the sliding
     * window when the kind is a sliding-window kind and the widths agree (vLLM fills the window only for its
     * sliding-window specs).
     */
    Spec specFor(int[] pages) {
        String kind = null;
        boolean mixed = false;
        int window = -1;
        for (int g = 0; g < pages.length; g++) {
            if (pages[g] < 0) {
                continue;
            }
            if (kind == null) {
                kind = groupKinds[g];
                window = groupWindows[g];
            } else if (!kind.equals(groupKinds[g]) || groupWindows[g] != window) {
                mixed = true;
            }
        }
        if (mixed || kind == null) {
            // no page-holding group (or disagreeing ones): describe the whole cache or nothing
            kind = allGroupsKind();
            window = sharedWindow();
        }
        if (kind == null) {
            return null;
        }
        boolean sliding = kind.equals("sliding_window") || kind.equals("sliding_window_mla");
        return new Spec(kind, sliding && window > 0 ? Integer.valueOf(window) : null);
    }

    /** One KV cache spec: the kind of its group and the sliding window of a sliding-window group. */
    static final class Spec {
        final String kind;
        final Integer window;

        Spec(String kind, Integer window) {
            this.kind = kind;
            this.window = window;
        }
    }

    /** The groups' shared kind, or null when they disagree or the memory has no KV group. */
    private String allGroupsKind() {
        String kind = null;
        for (String k : groupKinds) {
            if (k == null) {
                continue;
            }
            if (kind == null) {
                kind = k;
            } else if (!kind.equals(k)) {
                return null;
            }
        }
        return kind;
    }

    /** The groups' shared sliding window, 0 for none or when they disagree. */
    private int sharedWindow() {
        int window = -1;
        for (int w : groupWindows) {
            if (window < 0) {
                window = w;
            } else if (window != w) {
                return 0;
            }
        }
        return Math.max(window, 0);
    }

    // ------------------------------------------------------------------ lookup and adoption

    /**
     * The longest cached prefix of {@code rows} under {@code key}, at most {@code maxTokens} rows long and never
     * ending inside a media item: whole blocks along the hash chain, then the cached block (whole or partial)
     * sharing the longest row prefix with what follows, the most recently used winning ties.
     */
    Match match(CacheRows rows, PrefixKey key, int maxTokens) {
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(key, "key");
        validate();
        int limit = Math.min(maxTokens, rows.length());
        if (limit <= 0) {
            return Match.NONE;
        }
        List<Block> hits = new ArrayList<>();
        Block parent = null;
        int pos = 0;
        while (pos + pageSize <= limit) {
            Block b = find(parent, rows, pos, pageSize, key);
            if (b == null || !b.full) {
                break;
            }
            hits.add(b);
            parent = b;
            pos += pageSize;
        }
        if (checkpoints != null) {
            return restorable(hits, siblings(parent, key), rows, pos, limit);
        }
        Block best = null;
        int bestLen = 0;
        for (Block c : siblings(parent, key)) {
            int len = CacheRows.commonLength(c.rows, 0, rows, pos, Math.min(c.rows.length(), limit - pos));
            if (len > bestLen || (len == bestLen && len > 0 && c.lastUse > best.lastUse)) {
                best = c;
                bestLen = len;
            }
        }
        return truncated(hits, best, bestLen, rows.cut(hits.size() * pageSize + bestLen));
    }

    /**
     * The longest restorable prefix of {@code rows}: the context checkpoint, among those hanging on the whole
     * blocks {@code hits} and on the non-full blocks beside them — a partial child the shared rows cover exactly
     * (the checkpoint's state was taken at rows the adopting prompt must match row for row), and the {@code
     * boundary} blocks after them (as far as each shares the rows from {@code pos}) — that restores the most rows
     * within {@code limit} without ending inside a media item, the most recently used winning ties; the match
     * ends at its rows.
     */
    private Match restorable(List<Block> hits, List<Block> boundary, CacheRows rows, int pos, int limit) {
        ContextCheckpoint best = null;
        for (Block b : hits) {
            best = better(best, b.checkpoints, rows, limit);
            for (Block child : b.children) {
                if (child.full) {
                    continue;
                }
                int from = child.start;
                int len = CacheRows.commonLength(child.rows, 0, rows, from,
                        Math.min(child.rows.length(), limit - from));
                for (ContextCheckpoint c : child.checkpoints) {
                    if (c.rows - from <= len) {
                        best = better(best, c, rows, limit);
                    }
                }
            }
        }
        for (Block b : boundary) {
            int len = CacheRows.commonLength(b.rows, 0, rows, pos, Math.min(b.rows.length(), limit - pos));
            for (ContextCheckpoint c : b.checkpoints) {
                if (c.rows - pos <= len) {
                    best = better(best, c, rows, limit);
                }
            }
        }
        if (best == null) {
            return Match.NONE;
        }
        int whole = best.rows / pageSize;
        int rest = best.rows % pageSize;
        return new Match(new ArrayList<>(hits.subList(0, whole)), rest > 0 ? best.block : null, rest, pageSize,
                best);
    }

    /** The checkpoint of {@code candidates} that restores the most rows of {@code rows} within {@code limit}. */
    private static ContextCheckpoint better(ContextCheckpoint best, List<ContextCheckpoint> candidates,
                                            CacheRows rows, int limit) {
        for (ContextCheckpoint c : candidates) {
            best = better(best, c, rows, limit);
        }
        return best;
    }

    /** {@code candidate} when it restores within {@code limit} rows outside a media item and beats {@code best}. */
    private static ContextCheckpoint better(ContextCheckpoint best, ContextCheckpoint candidate, CacheRows rows,
                                            int limit) {
        if (candidate.rows > limit || rows.cut(candidate.rows) != candidate.rows) {
            return best;
        }
        if (best == null || candidate.rows > best.rows
                || (candidate.rows == best.rows && candidate.lastUse > best.lastUse)) {
            return candidate;
        }
        return best;
    }

    /**
     * The match of the whole blocks {@code hits} and {@code bestLen} rows of the boundary block {@code best}, cut
     * back to its first {@code keep} rows: the blocks before the cut, then the block the cut falls inside.
     */
    private Match truncated(List<Block> hits, Block best, int bestLen, int keep) {
        int whole = keep / pageSize;
        int rest = keep % pageSize;
        if (whole >= hits.size()) {
            return new Match(hits, rest > 0 ? best : null, rest, pageSize, null);
        }
        List<Block> blocks = new ArrayList<>(hits.subList(0, whole));
        return new Match(blocks, rest > 0 ? hits.get(whole) : null, rest, pageSize, null);
    }

    /**
     * Gives the empty sequence {@code seq} the matched blocks' pages as its first {@code match.tokens()} rows, at
     * the positions {@code rows} stores them ({@link PagedSequences#adoptPages}, its media and prompt-embedding rows
     * recorded as embedding rows), marks them used, and starts {@code chain} at the last whole block. A match ending
     * at a context checkpoint also gives the sequence the checkpoint's window pages and writes its recurrent state
     * into the sequence's state row. A free boundary leaf the adopter would copy on its first write while some KV
     * group has no idle page for the copy passes to the adopter instead: the pool drops it ({@code BlockRemoved})
     * and the adopter writes the page in place, as eviction would have freed that very page for the copy.
     */
    void adopt(int seq, Match match, CacheRows rows, BlockChain chain) {
        boolean takeOver = match.partial != null && match.partial.children.isEmpty()
                && !paged.pagesShared(match.partial.pages);
        int[][] pages = new int[match.blocks.size() + (match.partial != null ? 1 : 0)][];
        for (int k = 0; k < match.blocks.size(); k++) {
            pages[k] = match.blocks.get(k).pages;
        }
        int lastRows = pageSize;
        if (match.partial != null) {
            pages[pages.length - 1] = match.partial.pages;
            lastRows = match.partialRows;
        }
        int n = match.tokens();
        if (match.checkpoint != null) {
            paged.adoptPages(seq, pages, lastRows, rows.tokens(0, n), rows.positions(0, n),
                    match.checkpoint.windows(), embeddingToken);
            checkpoints.restore(seq, match.checkpoint);
            touch(match.checkpoint);
        } else {
            paged.adoptPages(seq, pages, lastRows, rows.tokens(0, n), rows.positions(0, n), embeddingToken);
        }
        for (Block b : match.blocks) {
            touch(b);
        }
        if (takeOver && paged.minFreePages() == 0) {
            evict(match.partial);
        } else if (match.partial != null) {
            touch(match.partial);
        }
        chain.tail = match.blocks.isEmpty() ? null : match.blocks.get(match.blocks.size() - 1);
        chain.blocks = match.blocks.size();
        chain.stopped = false;
    }

    /** The cached block after {@code parent} holding exactly the rows {@code [from, from + len)}, or null. */
    Block find(Block parent, CacheRows rows, int from, int len, PrefixKey key) {
        byte[] h = hashOf(parent, rows, from, len, parent == null ? key : null);
        long mapKey = KvCacheEvent.hashInt(h);
        for (Block b = byHash.get(mapKey); b != null; b = b.nextSameHash) {
            if (b.parent == parent && b.rows.length() == len && (parent != null || b.key.equals(key))
                    && Arrays.equals(b.hash, h)
                    && CacheRows.commonLength(b.rows, 0, rows, from, len) == len) {
                return b;
            }
        }
        return null;
    }

    /** The cached blocks after {@code parent}: its children, or the chain roots of {@code key}. */
    List<Block> siblings(Block parent, PrefixKey key) {
        if (parent != null) {
            return parent.children;
        }
        List<Block> out = new ArrayList<>();
        for (Block r : roots) {
            if (r.key.equals(key)) {
                out.add(r);
            }
        }
        return out;
    }

    /**
     * The cached blocks that extend {@code parent} in its chain ({@code parent}'s children, or the chain roots of
     * {@code key} for null), the set the KV chunk reuse's scan walks past a match boundary. Only cached blocks are
     * listed: eviction unlists a block from its parent.
     */
    List<Block> chainChildren(Block parent, PrefixKey key) {
        return siblings(parent, key);
    }

    // ------------------------------------------------------------------ storing

    /**
     * Stores the whole blocks of {@code seq}'s first {@code rows} committed rows ({@code keyed} holds at least
     * that many) that {@code chain} does not cover yet: a block another history already cached under the same
     * chain extends the chain as it is, a new one takes the pool's reference on its pages. Each run of newly
     * stored blocks queues one {@code BlockStored}. Storing stops for good once the sequence's rows no longer map
     * onto block positions, or the chain's tail left the cache.
     */
    void commit(int seq, BlockChain chain, CacheRows keyed, int rows, PrefixKey key) {
        validate();
        if (chain.stopped || (chain.tail != null && !chain.tail.cached)) {
            chain.stopped = true;
            return;
        }
        List<Block> run = new ArrayList<>();
        Block runParent = chain.tail;
        while ((chain.blocks + 1) * pageSize <= rows) {
            int from = chain.blocks * pageSize;
            Block b = find(chain.tail, keyed, from, pageSize, key);
            if (b == null) {
                int[] pages = paged.blockPages(seq, chain.blocks, pageSize, keyed.positions(from, from + pageSize));
                if (pages == null) {
                    chain.stopped = true;
                    break;
                }
                b = store(chain.tail, keyed, from, pageSize, true, key, pages);
                run.add(b);
            } else {
                storedEvent(runParent, run);
                run.clear();
                runParent = b;
                touch(b);
            }
            chain.tail = b;
            chain.blocks++;
        }
        storedEvent(runParent, run);
    }

    /**
     * Stores the trailing partial block of {@code seq}'s {@code rows} committed rows after {@link #commit} covered
     * its whole blocks, and returns the cached block that now holds those tokens (stored, or an existing block
     * starting with them), or null when there is none or the rows do not map onto block positions. Called when
     * the sequence stops writing (retirement, preemption): a partial block never extends further.
     */
    Block commitPartial(int seq, BlockChain chain, CacheRows keyed, int rows, PrefixKey key) {
        int from = chain.blocks * pageSize;
        int m = rows - from;
        if (chain.stopped || m <= 0 || m >= pageSize || (chain.tail != null && !chain.tail.cached)) {
            return null;
        }
        for (Block c : siblings(chain.tail, key)) {
            if (c.rows.length() >= m && CacheRows.commonLength(c.rows, 0, keyed, from, m) == m) {
                touch(c);
                return c;
            }
        }
        int[] pages = paged.blockPages(seq, chain.blocks, m, keyed.positions(from, rows));
        if (pages == null) {
            return null;
        }
        Block b = store(chain.tail, keyed, from, m, false, key, pages);
        storedEvent(chain.tail, Collections.singletonList(b));
        return b;
    }

    /**
     * Hangs the context checkpoint of sequence {@code seq} at its {@code rows} committed rows on the cached block
     * {@code block}, which holds row {@code rows - 1}; {@code owner} names the run that took it. A checkpoint
     * already hanging there at the same rows is kept (the same chain of rows computes the same state) and marked
     * used. Beyond the checkpoint limit, llama.cpp's eviction rules make room first: while the list is nearly
     * full, a checkpoint of another run that sits within {@link ContextCheckpoints#minStep()} rows of the kept
     * checkpoint before it leaves (its rows re-prefill cheaply, a spread-out one stays), then the oldest created
     * checkpoints leave until the list fits.
     *
     * @throws IllegalArgumentException when row {@code rows - 1} lies outside {@code block}
     */
    void checkpoint(Block block, int seq, int rows, Object owner) {
        validate();
        if (checkpoints == null || block == null || !block.cached) {
            return;
        }
        if (rows <= block.start || rows > block.start + block.rows.length()) {
            throw new IllegalArgumentException("a checkpoint at " + rows + " rows does not end inside the block of"
                    + " rows [" + block.start + ", " + (block.start + block.rows.length()) + ")");
        }
        for (ContextCheckpoint c : block.checkpoints) {
            if (c.rows == rows) {
                touch(c);
                return;
            }
        }
        evictCloseCheckpoints(owner);
        ContextCheckpoint c = checkpoints.capture(seq, rows);
        c.block = block;
        c.owner = owner;
        block.checkpoints.add(c);
        touch(c);
        checkpointAge.add(c);
        while (checkpointLru.size() > checkpoints.limit()) {
            drop(checkpointAge.get(0));
        }
    }

    /**
     * Drops the checkpoints the min-step rule erases for a capture by {@code owner} (llama.cpp's
     * {@code create_checkpoint}): while the list plus the capture fills the limit, every checkpoint of another
     * owner within {@code minStep} rows of the kept checkpoint before it leaves, oldest first; a checkpoint of
     * {@code owner} and the first of the walk stay. A no-op when {@code minStep} is 0 or the list has room.
     */
    private void evictCloseCheckpoints(Object owner) {
        int minStep = checkpoints.minStep();
        if (minStep <= 0 || checkpointLru.size() + 1 < checkpoints.limit()) {
            return;
        }
        int last = -1;
        for (Iterator<ContextCheckpoint> it = checkpointAge.iterator();
                checkpointLru.size() + 1 >= checkpoints.limit() && it.hasNext(); ) {
            ContextCheckpoint c = it.next();
            if (sameOwner(c, owner) || last < 0 || c.rows > last + minStep) {
                last = c.rows;
                continue;
            }
            it.remove();
            drop(c);
        }
    }

    /** Whether {@code c} belongs to the run {@code owner}, which the min-step eviction spares. */
    private static boolean sameOwner(ContextCheckpoint c, Object owner) {
        return owner != null && owner == c.owner;
    }

    /**
     * Drops least recently used context checkpoints until sliding-window group {@code g} has {@code pages} idle
     * pages, sparing those hanging on a block of {@code keep} and passing over those whose release would free none
     * of the group's pages (a running sequence still reads them). False when the group still lacks them.
     */
    boolean yieldWindowPages(int g, int pages, Set<Block> keep) {
        List<ContextCheckpoint> lru = new ArrayList<>(checkpointLru);
        for (ContextCheckpoint c : lru) {
            if (paged.freePages(g) >= pages) {
                break;
            }
            PageWindow w = c.windows()[g];
            if (w != null && !keep.contains(c.block) && paged.windowOnlyPages(w) > 0) {
                drop(c);
            }
        }
        return paged.freePages(g) >= pages;
    }

    /** Takes {@code c} off its block and the pool and releases its window pages. */
    private void drop(ContextCheckpoint c) {
        checkpointLru.remove(c);
        checkpointAge.remove(c);
        c.block.checkpoints.remove(c);
        checkpoints.release(c);
    }

    private void touch(ContextCheckpoint c) {
        checkpointLru.remove(c);
        checkpointLru.add(c);
        c.lastUse = ++clock;
    }

    /**
     * Records that a sequence released its pages: its partial block, then its chain from tail to root, become
     * the most recently used blocks, the root newest, so the tail goes before its prefix (vLLM frees a request's
     * blocks in reverse order).
     */
    void released(BlockChain chain, Block partial) {
        if (partial != null && partial.cached) {
            touch(partial);
        }
        for (Block b = chain.tail; b != null; b = b.parent) {
            if (b.cached) {
                touch(b);
            }
        }
    }

    /**
     * Evicts the blocks of {@code chain} from block {@code cutBlocks} on and rewinds the chain to its first
     * {@code cutBlocks} blocks: a context shift moves the rows those blocks keyed, while the kept prefix below
     * the cut stays cached and the chain stays extendable over the shifted rows (the next {@link #commit} stores
     * the seam block over the post-shift rows). The evicted blocks' children — other chains sharing the prefix —
     * stay cached with their pages; their own commits stop at their uncached tail.
     */
    void rewind(BlockChain chain, int cutBlocks) {
        if (cutBlocks >= chain.blocks) {
            return;
        }
        List<Block> doomed = new ArrayList<>();
        Block tail = chain.tail;
        for (int left = chain.blocks - cutBlocks; left > 0 && tail != null; left--) {
            doomed.add(tail);
            tail = tail.parent;
        }
        for (Block b : doomed) {
            evict(b);
        }
        chain.tail = tail;
        chain.blocks = cutBlocks;
    }

    /**
     * Inserts the block over rows {@code [from, from + len)} of {@code keyed} after {@code parent} over
     * {@code pages} (taking the pool's reference) and evicts the free partial siblings whose rows it extend: they
     * serve nothing it does not — except a partial holding context checkpoints, whose state no whole block of
     * another run restores, so it stays.
     */
    private Block store(Block parent, CacheRows keyed, int from, int len, boolean full, PrefixKey key,
                        int[] pages) {
        PrefixKey chainKey = parent == null ? key : null;
        byte[] h = hashOf(parent, keyed, from, len, chainKey);
        List<BlockExtraKey> keys = keyed.extraKeys(from, from + len, chainKey);
        paged.holdPages(pages);
        CacheRows rows = keyed.slice(from, from + len);
        Block b = new Block(h, parent, rows, full, chainKey, pages, keys.isEmpty() ? null : keys);
        b.nextSameHash = byHash.get(b.mapKey);
        byHash.put(b.mapKey, b);
        List<Block> dominated = new ArrayList<>();
        for (Block c : siblings(parent, key)) {
            int n = c.rows.length();
            if (!c.full && n < len && CacheRows.commonLength(c.rows, 0, rows, 0, n) == n
                    && !paged.pagesShared(c.pages) && c.checkpoints.isEmpty()) {
                dominated.add(c);
            }
        }
        if (parent == null) {
            roots.add(b);
        } else {
            parent.children.add(b);
        }
        touch(b);
        for (Block c : dominated) {
            evict(c);
        }
        return b;
    }

    /**
     * Queues a {@code BlockStored} for the whole blocks {@code match} reuses from the cache, as vLLM's
     * {@code emit_cached_block_events} does for a request whose {@code kv_cache_report_mode} is {@code full}: the
     * same hashes, tokens, extra keys, adapters and cache spec their store carried, the first block extending no
     * parent (a cache hit starts at its chain's first block). The boundary block a match ends inside is not
     * reported; it holds no whole block's worth of the prompt's rows.
     */
    void reportReused(Match match) {
        Objects.requireNonNull(match, "match");
        if (match.blocks.isEmpty()) {
            return;
        }
        storedEvent(match.blocks.get(0).parent, match.blocks);
    }

    private void storedEvent(Block parent, List<Block> run) {
        if (!events || run.isEmpty()) {
            return;
        }
        int size = run.get(0).rows.length();
        byte[][] hashes = new byte[run.size()][];
        int[] tokens = new int[run.size() * size];
        List<List<BlockExtraKey>> extra = new ArrayList<>(run.size());
        for (int i = 0; i < run.size(); i++) {
            Block b = run.get(i);
            hashes[i] = b.hash;
            System.arraycopy(b.rows.tokens(0, size), 0, tokens, i * size, size);
            extra.add(b.extraKeys);
        }
        PrefixKey chain = chainKey(run.get(0));
        Spec spec = specFor(run.get(0).pages);
        pending.add(new KvCacheEvent.BlockStored(hashes, parent == null ? null : parent.hash, tokens, size,
                medium, chain == null ? null : chain.loraId(), chain == null ? null : chain.loraName(),
                spec == null ? null : spec.kind, spec == null ? null : spec.window, extra));
    }

    /** Whether KV cache events are queued (the tiers consult it before building one). */
    boolean eventsOn() {
        return events;
    }

    /** Queues a KV cache event for {@link #takeEvents} to hand the publisher. */
    void queueEvent(KvCacheEvent e) {
        pending.add(e);
    }

    /** The extra keys of the chain {@code b} belongs to: those of its first block. */
    static PrefixKey chainKey(Block b) {
        Block root = b;
        while (root.parent != null) {
            root = root.parent;
        }
        return root.key;
    }

    private void touch(Block b) {
        lru.remove(b);
        lru.add(b);
        b.lastUse = ++clock;
    }

    // ------------------------------------------------------------------ host images

    /**
     * Installs {@code image} as the cached block after {@code parent} under the chain key {@code key} (a restored
     * slot's block, or a host-RAM block coming back): a cached block already holding its rows serves as it is;
     * otherwise free cached blocks outside {@code keep} are evicted until every KV group has an idle page, the rows
     * are written into fresh pages and the block is stored with the pool's reference, queuing its
     * {@code BlockStored}.
     *
     * @return the cached block holding the image's rows
     * @throws IllegalStateException when no cached block can be evicted to make room
     */
    Block install(Block parent, BlockImage image, PrefixKey key, BlockLayout layout, Set<Block> keep) {
        validate();
        CacheRows rows = CacheRows.ofTokens(image.tokens());
        int n = rows.length();
        Block existing = image.full() ? find(parent, rows, 0, n, key) : holding(parent, rows, n, key);
        if (existing != null) {
            touch(existing);
            return existing;
        }
        if (!makeRoom(keep)) {
            throw new IllegalStateException("No available space in KV cache: every page of some KV group holds"
                    + " rows in use (" + paged + ")");
        }
        int[] pages = paged.importPages(image.positions());
        try {
            image.install(layout, pages);
        } catch (RuntimeException e) {
            paged.releasePages(pages);
            throw e;
        }
        Block b = store(parent, rows, 0, n, image.full(), key, pages);
        paged.releasePages(pages);
        storedEvent(parent, Collections.singletonList(b));
        return b;
    }

    /**
     * Evicts free cached blocks outside {@code keep} until every KV group has an idle page; false when none is left
     * to evict.
     */
    boolean makeRoom(Set<Block> keep) {
        while (paged.minFreePages() == 0) {
            if (!evictOne(keep)) {
                return false;
            }
        }
        return true;
    }

    /** A cached block after {@code parent} whose rows start with the first {@code m} of {@code rows}, or null. */
    private Block holding(Block parent, CacheRows rows, int m, PrefixKey key) {
        for (Block c : siblings(parent, key)) {
            if (c.rows.length() >= m && CacheRows.commonLength(c.rows, 0, rows, 0, m) == m) {
                return c;
            }
        }
        return null;
    }

    /**
     * Evicts {@code last} and then its ancestors while each is a free cached leaf outside {@code spare}: the blocks
     * only the erased history used return to the pools, a prefix another chain extends stays.
     */
    void evictChain(Block last, Set<Block> spare) {
        validate();
        for (Block b = last; b != null; ) {
            Block parent = b.parent;
            if (!b.cached || !b.children.isEmpty() || spare.contains(b) || paged.pagesShared(b.pages)) {
                return;
            }
            evict(b);
            b = parent;
        }
    }

    // ------------------------------------------------------------------ offload tiers

    /**
     * Starts the background reads a later adoption of the prefix {@code rows[0, maxTokens)} under {@code key} would
     * need from the disk tier, and answers whether every block the disk can contribute is already in memory; true
     * without a disk tier ({@link BlockTiers#prefetch}).
     */
    boolean prefetch(CacheRows rows, PrefixKey key, int maxTokens) {
        return tiers.prefetch(rows, key, maxTokens);
    }

    /**
     * Brings back the host-RAM blocks extending the cached prefix of {@code rows} under {@code key}, at most
     * {@code maxTokens} rows deep, so the next {@link #match} finds them in the memory ({@link
     * BlockTiers#promote}).
     *
     * @return the blocks brought back
     */
    int promote(CacheRows rows, PrefixKey key, int maxTokens) {
        return tiers.promote(rows, key, maxTokens);
    }

    /**
     * Host copies of the cached prefix of the token rows {@code rows} under {@code key}, at most {@code maxTokens}
     * rows: whole blocks along the chain, then the boundary block the memory, the host or the disk shares most of.
     * Reads only; nothing moves.
     *
     * @throws UnsupportedOperationException when a block holds media or prompt-embedding rows
     */
    List<BlockImage> images(CacheRows rows, PrefixKey key, int maxTokens) {
        return tiers.images(rows, key, maxTokens);
    }

    /**
     * Drops the host-RAM and disk blocks whose whole chain lies within the first {@code count} rows of {@code rows}
     * under {@code key}, except those {@code spare} answers true for ({@link BlockTiers#forgetHost}).
     */
    void forgetHost(CacheRows rows, PrefixKey key, int count, Predicate<HostBlockCache.Entry> spare) {
        tiers.forgetHost(rows, key, count, spare);
    }


    // ------------------------------------------------------------------ eviction

    /**
     * Evicts the least recently used free leaf block outside {@code keep}: its pages return to the pools. False
     * when no cached block is free (every one is in use, has cached children or is kept).
     */
    boolean evictOne(Set<Block> keep) {
        validate();
        if (checkpoints != null && evictFirst(keep, true)) {
            return true;
        }
        return evictFirst(keep, false);
    }

    /**
     * Evicts the least recently used free leaf outside {@code keep} — only one without checkpoints when
     * {@code bare} is set; false when there is none.
     */
    private boolean evictFirst(Set<Block> keep, boolean bare) {
        for (Block b : lru) {
            if (b.children.isEmpty() && (keep == null || !keep.contains(b)) && !paged.pagesShared(b.pages)
                    && (!bare || b.checkpoints.isEmpty())) {
                tiers.offload(b);
                evict(b);
                return true;
            }
        }
        return false;
    }

    /**
     * Evicts least recently used free leaf blocks while the free blocks hold more tokens than the budget (no-op
     * without one).
     */
    void trimToBudget() {
        if (budget <= 0) {
            return;
        }
        long free = 0;
        for (Block b : lru) {
            if (!paged.pagesShared(b.pages)) {
                free += b.rows.length();
            }
        }
        while (free > budget) {
            Block victim = null;
            for (Block b : lru) {
                if (b.children.isEmpty() && !paged.pagesShared(b.pages)) {
                    victim = b;
                    break;
                }
            }
            if (victim == null) {
                return;
            }
            free -= victim.rows.length();
            tiers.offload(victim);
            evict(victim);
        }
    }

    private void evict(Block b) {
        for (ContextCheckpoint c : new ArrayList<>(b.checkpoints)) {
            drop(c);
        }
        unlink(b);
        if (b.parent == null) {
            roots.remove(b);
        } else {
            b.parent.children.remove(b);
        }
        lru.remove(b);
        b.cached = false;
        paged.releasePages(b.pages);
        if (events) {
            pending.add(new KvCacheEvent.BlockRemoved(new byte[][] {b.hash}, medium));
        }
    }

    private void unlink(Block b) {
        Block head = byHash.get(b.mapKey);
        if (head == b) {
            if (b.nextSameHash == null) {
                byHash.remove(b.mapKey);
            } else {
                byHash.put(b.mapKey, b.nextSameHash);
            }
            return;
        }
        for (Block prev = head; prev != null; prev = prev.nextSameHash) {
            if (prev.nextSameHash == b) {
                prev.nextSameHash = b.nextSameHash;
                return;
            }
        }
    }

    /**
     * Drops every cached block, releasing the pool's page references, and the host-RAM and disk tiers, and queues
     * {@code AllBlocksCleared} (vLLM's {@code reset_prefix_cache}).
     */
    void clear() {
        for (ContextCheckpoint c : new ArrayList<>(checkpointLru)) {
            drop(c);
        }
        for (Iterator<Block> it = lru.iterator(); it.hasNext(); ) {
            Block b = it.next();
            b.cached = false;
            paged.releasePages(b.pages);
        }
        lru.clear();
        byHash.clear();
        roots.clear();
        tiers.clear();
        if (events) {
            pending.add(KvCacheEvent.AllBlocksCleared.INSTANCE);
        }
    }

    /** Drops the pool when the memory was cleared under it: its pages' rows no longer serve attention. */
    void validate() {
        if (paged.clears() != clears) {
            clears = paged.clears();
            clear();
        }
    }

    /** The events queued since the last call, in order; the queue empties. Drains the disk tier first. */
    List<KvCacheEvent> takeEvents() {
        tiers.pumpDisk();
        if (pending.isEmpty()) {
            return Collections.emptyList();
        }
        List<KvCacheEvent> out = new ArrayList<>(pending);
        pending.clear();
        return out;
    }

    /**
     * Waits at most {@code millis} for the disk tier's admitted writes to become durable and queues their events;
     * true when every write finished. A drain uses it so a finished engine still leaves its spills on the disk.
     */
    boolean awaitDisk(long millis) {
        return tiers.awaitDisk(millis);
    }

    /**
     * Stops the disk tier's background workers, letting the writes it accepted finish for a bounded time and
     * abandoning the rest, then drains what completed. The engine's states stay as they are.
     */
    void close() {
        tiers.close();
    }

    @Override
    public String toString() {
        return "PrefixCache[" + lru.size() + " blocks of " + pageSize + ", budget " + budget
                + (checkpoints == null ? "" : ", " + checkpointLru.size() + " checkpoints") + "]";
    }
}
