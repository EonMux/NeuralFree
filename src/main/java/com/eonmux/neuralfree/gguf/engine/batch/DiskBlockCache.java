package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The disk tier of the prefix cache (vLLM's tiering filesystem tier, llama.cpp's {@code --prompt-cache} file): the
 * blocks the host-RAM tier ({@link HostBlockCache}) dropped for its budget, kept as one file per block under a
 * directory, least recently used first out of a byte budget, so a prompt cache outlives the process and survives an
 * engine restart. A file is named by its block's chain hash and its parent's hash — the block hasher's digests
 * ({@link BlockHasher}, vLLM's {@code hash_block_tokens}), hex — under one fan-out directory of the parent
 * digest's leading bytes, which keeps the blocks of a chain together: a lookup enumerates the parent's children,
 * reads each candidate and verifies the whole chain of tokens, the extra keys and the recomputed chain hash before
 * it answers, so a foreign or corrupted file can only miss, never serve wrong rows.
 *
 * <p>The tier's files live in one namespace directory per (block layout, hasher identity): the layout identity (page
 * size, cache dtypes and widths) and the hasher's identity (algorithm name and its {@code NONE_HASH} digest — a
 * digest of the seed, never the seed itself) name it, so another model's files and the files of another hash
 * algorithm or another {@code PYTHONHASHSEED} seed sit in another namespace, are never indexed by this one and can
 * only miss, never be served or deleted — exactly another layout's files. Only a file inside this namespace that
 * fails its own verification (a truncated or corrupted one) is deleted. Writes land through a temp file and an
 * atomic move. I/O failures never fail a request: a write that fails drops the block from the tier, a read that
 * fails is a miss.
 *
 * <p>The tier services both directions off the engine's thread, as vLLM's filesystem tier runs its stores and loads
 * on a worker pool. {@link #put} only serializes the block and queues it on a bounded background writer; a full
 * queue drops the spill rather than stalling the engine. A queued block stays promotable from its host image
 * ({@link #wholeLoaded}, {@link #bestLoaded}) until its file is durable, and only then is it indexed and published
 * ({@link #pump} answers the stored and evicted blocks in order, so a {@code BlockStored} on the storage medium
 * follows its file). {@link #prefetchReady} walks the chain the index knows, submits the files that are not in
 * memory to a background reader and answers whether every block a promotion would need is loaded; the scheduler
 * delays a request until it is, and a promotion that arrives earlier misses and falls back to recomputing the rows.
 * Loaded blocks stay in a bounded memory image ({@link #DEFAULT_RESIDENT_BYTES}) until a promotion takes them.
 * {@link #close()} drains the accepted writes for a bounded time, then abandons the rest and stops both workers, so
 * shutdown never leaves a writer running. Not thread-safe on the engine side: the engine thread owns the index and
 * calls every method, while the writer and reader threads only touch the files and the queues.
 */
final class DiskBlockCache {

    private static final Logger LOGGER = LogManager.getLogger(DiskBlockCache.class);

    private static final int MAGIC = 0x4E464B42; // "NFKB"
    private static final int VERSION = 3;
    private static final String SUFFIX = ".bin";
    /** The chain length a file may claim; anything above is garbage, the context is far below it. */
    private static final int MAX_CHAIN = 1 << 26;
    /** Hex characters of the namespace digest that name the namespace directory. */
    private static final int NAMESPACE_HEX = 16;
    /** Accepted writes the background writer may hold before a spill is dropped instead of queued. */
    private static final int DEFAULT_WRITE_QUEUE = 32;
    /** Threads that read the blocks a promotion needs. */
    private static final int DEFAULT_READ_THREADS = 2;
    /** Loaded blocks kept in memory for promotion when the tier's byte budget is unbounded. */
    private static final long DEFAULT_RESIDENT_BYTES = 128L << 20;
    /** Distinguishes the temp files of concurrent writers without a blocking entropy source. */
    private static final AtomicLong TEMP_FILES = new AtomicLong();
    /** How long the accepted writes may run at shutdown before they are abandoned. */
    private static final long SHUTDOWN_DRAIN_MILLIS = 5000;
    /** How long an operation that must leave no file behind waits for an in-flight write. */
    private static final long BLOCKING_DRAIN_MILLIS = 10000;
    /** Completed prefetches remembered before the oldest are forgotten. */
    private static final int MAX_PREFETCHES = 256;

    /** The file operations of the tier; a slow implementation proves the engine never waits on them. */
    interface Io {
        byte[] read(Path file) throws IOException;

        void write(byte[] blob, Path file) throws IOException;

        void delete(Path file) throws IOException;
    }

    /** The real filesystem. */
    static final Io FILES = new Io() {
        @Override
        public byte[] read(Path file) throws IOException {
            return Files.readAllBytes(file);
        }

        @Override
        public void write(byte[] blob, Path file) throws IOException {
            writeFile(blob, file);
        }

        @Override
        public void delete(Path file) throws IOException {
            Files.deleteIfExists(file);
        }
    };

    /** One result of {@link #put}: whether the entry was accepted, and the hashes a synchronous tier would drop. */
    static final class Spill {
        static final Spill NOT_STORED = new Spill(false, Collections.emptyList());

        final boolean stored;
        /** Empty: the budget's drops are reported by {@link #pump} once the write is durable. */
        final List<byte[]> evicted;

        Spill(boolean stored, List<byte[]> evicted) {
            this.stored = stored;
            this.evicted = evicted;
        }
    }

    /** What the background work did since the last {@link #pump}: a durable store, or a dropped block file. */
    static final class Event {
        /** The entry the writer made durable, or null for a removal. */
        final HostBlockCache.Entry stored;
        /** The digest of the block whose file left the tier, or null for a store. */
        final byte[] removed;

        private Event(HostBlockCache.Entry stored, byte[] removed) {
            this.stored = stored;
            this.removed = removed;
        }

        static Event stored(HostBlockCache.Entry entry) {
            return new Event(entry, null);
        }

        static Event removed(byte[] hash) {
            return new Event(null, hash);
        }
    }

    /** The index of one block file: its chain digests, its path and its size on disk. */
    private static final class Ref {
        final byte[] hash;
        final byte[] parentHash;
        /** The value-equality key {@code parentHash} sits under in the children map. */
        private final HostBlockCache.DigestKey parentKey;
        final Path file;
        final long bytes;

        Ref(byte[] hash, byte[] parentHash, Path file, long bytes) {
            this.hash = hash;
            this.parentHash = parentHash;
            this.parentKey = new HostBlockCache.DigestKey(parentHash);
            this.file = file;
            this.bytes = bytes;
        }
    }

    /** A block accepted by {@link #put} and not yet durable; its host image serves promotions meanwhile. */
    private static final class Pending {
        final HostBlockCache.Entry entry;
        final byte[] blob;
        final Path file;
        final long bytes;
        final HostBlockCache.DigestKey key;
        final HostBlockCache.DigestKey parentKey;
        volatile boolean cancelled;

        Pending(HostBlockCache.Entry entry, byte[] blob, Path file) {
            this.entry = entry;
            this.blob = blob;
            this.file = file;
            this.bytes = blob.length;
            this.key = new HostBlockCache.DigestKey(entry.hash);
            this.parentKey = new HostBlockCache.DigestKey(entry.parentHash);
        }
    }

    /** A writer job: a block to store, a file to delete, or the stop sentinel. */
    private static final class Write {
        static final Write STOP = new Write(null, null);

        final Pending pending;
        final Path delete;

        private Write(Pending pending, Path delete) {
            this.pending = pending;
            this.delete = delete;
        }

        static Write store(Pending pending) {
            return new Write(pending, null);
        }

        static Write delete(Path file) {
            return new Write(null, file);
        }
    }

    /** A completed writer job, read by the engine thread. */
    private static final class WriteDone {
        final Pending pending;
        final boolean ok;
        final IOException failure;

        WriteDone(Pending pending, boolean ok, IOException failure) {
            this.pending = pending;
            this.ok = ok;
            this.failure = failure;
        }
    }

    /** One file a prefetch reads. */
    private static final class ReadJob {
        final Ref ref;

        ReadJob(Ref ref) {
            this.ref = ref;
        }
    }

    /** The prefix a prefetch was asked for: one request's extra keys and its wanted token rows. */
    private static final class PrefetchKey {
        private final PrefixKey key;
        private final int[] tokens;

        PrefetchKey(PrefixKey key, int[] tokens) {
            this.key = key;
            this.tokens = tokens;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof PrefetchKey)) {
                return false;
            }
            PrefetchKey p = (PrefetchKey) o;
            return key.equals(p.key) && Arrays.equals(tokens, p.tokens);
        }

        @Override
        public int hashCode() {
            return 31 * key.hashCode() + Arrays.hashCode(tokens);
        }
    }

    /** The state of one prefix's background read. */
    private static final class Prefetch {
        volatile boolean done;
    }

    /** A completed background read: the verified entries and the files that failed verification. */
    private static final class Load {
        final PrefetchKey key;
        final List<HostBlockCache.Entry> entries;
        final List<Ref> broken;

        Load(PrefetchKey key, List<HostBlockCache.Entry> entries, List<Ref> broken) {
            this.key = key;
            this.entries = entries;
            this.broken = broken;
        }
    }

    /** The namespace directory the tier's files live in. */
    private final Path dir;
    private final long budget;
    private final BlockLayout layout;
    private final BlockHasher hasher;
    /** The hasher's {@code NONE_HASH}: a chain's first block's parent digest, and the file identity. */
    private final byte[] noneHash;
    private final int pageSize;
    private final Io io;
    /** The indexed blocks by digest, least recently used first in {@link #lru}. */
    private final Map<HostBlockCache.DigestKey, Ref> byHash = new HashMap<>();
    private final Map<HostBlockCache.DigestKey, List<Ref>> children = new HashMap<>();
    private final LinkedHashSet<Ref> lru = new LinkedHashSet<>();
    private long bytes;
    /** Accepted writes not yet durable, by digest; their images serve promotions until the file is durable. */
    private final Map<HostBlockCache.DigestKey, Pending> pending = new HashMap<>();
    private final Map<HostBlockCache.DigestKey, List<Pending>> pendingChildren = new HashMap<>();
    /** Verified entries read off the disk, by digest, and the images the tier keeps for promotion. */
    private final Map<HostBlockCache.DigestKey, HostBlockCache.Entry> loaded = new HashMap<>();
    private final Map<HostBlockCache.DigestKey, List<HostBlockCache.Entry>> loadedChildren = new HashMap<>();
    private final LinkedHashSet<HostBlockCache.DigestKey> loadedLru = new LinkedHashSet<>();
    private long loadedBytes;
    private final long residentBudget;
    private final BlockingQueue<Write> writes;
    private final ConcurrentLinkedQueue<WriteDone> written = new ConcurrentLinkedQueue<>();
    private final ExecutorService readers;
    private final ConcurrentLinkedQueue<Load> loadedBlocks = new ConcurrentLinkedQueue<>();
    private final Map<PrefetchKey, Prefetch> prefetches = new HashMap<>();
    /** Events shutdown produced after the last {@link #pump}, answered by the next one. */
    private final List<Event> deferred = new ArrayList<>();
    /** Admitted writes the writer has not finished: queued, taken, or executing; the drains wait for it to empty. */
    private final AtomicInteger unfinishedWrites = new AtomicInteger();
    private final Thread writer;
    private volatile boolean closed;

    /**
     * A tier of at most {@code budgetBytes} of block files under {@code dir} (negative leaves it unbounded), whose
     * reads verify against {@code layout} and the chain hashes {@code hasher} computes; the directory is created,
     * the layout-and-hasher namespace is created under it, and the blocks left by earlier runs of the same layout
     * and hasher are indexed in name order.
     *
     * @throws IOException when the directory cannot be created or read
     */
    DiskBlockCache(Path dir, long budgetBytes, BlockLayout layout, BlockHasher hasher) throws IOException {
        this(dir, budgetBytes, layout, hasher, FILES, DEFAULT_WRITE_QUEUE, DEFAULT_READ_THREADS);
    }

    /** {@link #DiskBlockCache(Path, long, BlockLayout, BlockHasher)} over {@code io}. */
    DiskBlockCache(Path dir, long budgetBytes, BlockLayout layout, BlockHasher hasher, Io io) throws IOException {
        this(dir, budgetBytes, layout, hasher, io, DEFAULT_WRITE_QUEUE, DEFAULT_READ_THREADS);
    }

    /**
     * A tier over {@code io} whose writer holds at most {@code writeQueue} spills and whose reads run on
     * {@code readThreads} background threads.
     *
     * @throws IOException when the directory cannot be created or read
     */
    DiskBlockCache(Path dir, long budgetBytes, BlockLayout layout, BlockHasher hasher, Io io, int writeQueue,
                   int readThreads) throws IOException {
        Path root = Files.createDirectories(dir);
        this.dir = Files.createDirectories(root.resolve(namespace(layout.identity(), hasher)));
        this.budget = budgetBytes;
        this.layout = layout;
        this.hasher = Objects.requireNonNull(hasher, "hasher");
        this.noneHash = hasher.noneHash();
        this.pageSize = layout.pageSize();
        this.io = Objects.requireNonNull(io, "io");
        this.residentBudget = budgetBytes < 0 ? DEFAULT_RESIDENT_BYTES : Math.max(budgetBytes, 1L);
        this.writes = new ArrayBlockingQueue<>(Math.max(1, writeQueue));
        this.readers = Executors.newFixedThreadPool(Math.max(1, readThreads), threads("nf-disk-read"));
        this.writer = threads("nf-disk-write").newThread(this::writeLoop);
        this.writer.start();
        scan();
        LOGGER.info("disk tier under {} ({}): {} blocks, {} bytes of {}", root, dir.getFileName(), byHash.size(),
                bytes, budget < 0 ? "unbounded" : budget);
    }

    /** A daemon thread factory named after the tier's workers. */
    private static ThreadFactory threads(String prefix) {
        AtomicInteger n = new AtomicInteger();
        return r -> {
            Thread t = new Thread(r, prefix + "-" + n.getAndIncrement());
            t.setDaemon(true);
            return t;
        };
    }

    /**
     * The namespace name of a layout and hasher: hex of a SHA-256 digest over the layout identity, the algorithm
     * name and the {@code NONE_HASH} digest (a digest of the seed — the seed itself never lands on the disk).
     */
    private static String namespace(int[] identity, BlockHasher hasher) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (DataOutputStream s = new DataOutputStream(out)) {
            for (int v : identity) {
                s.writeInt(v);
            }
            s.writeUTF(hasher.algo().algoName());
            s.write(hasher.noneHash());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(out.toByteArray());
            StringBuilder hex = new StringBuilder(NAMESPACE_HEX);
            for (int i = 0; i < NAMESPACE_HEX / 2; i++) {
                hex.append(Character.forDigit((digest[i] >> 4) & 0xf, 16))
                        .append(Character.forDigit(digest[i] & 0xf, 16));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("no SHA-256 MessageDigest", e);
        }
    }

    /** Indexes the block files earlier runs left, in name order. */
    private void scan() throws IOException {
        List<Ref> found = new ArrayList<>();
        try (DirectoryStream<Path> groups = Files.newDirectoryStream(dir)) {
            for (Path group : groups) {
                if (!Files.isDirectory(group)) {
                    continue;
                }
                try (DirectoryStream<Path> files = Files.newDirectoryStream(group, "*" + SUFFIX)) {
                    for (Path f : files) {
                        Ref r = parse(f.getFileName().toString(), f);
                        if (r != null) {
                            found.add(r);
                        }
                    }
                }
            }
        }
        found.sort((a, b) -> a.file.compareTo(b.file));
        for (Ref r : found) {
            index(r);
        }
    }

    /**
     * The ref of a block file named {@code <hash hex>_<parent hex>.bin} — the hex of two chain digests — or null
     * for another name.
     */
    private static Ref parse(String name, Path file) {
        String base = name.substring(0, name.length() - SUFFIX.length());
        int at = base.indexOf('_');
        if (at < 0 || at == 0 || at + 1 >= base.length()) {
            return null;
        }
        try {
            byte[] hash = unhex(base.substring(0, at));
            byte[] parent = unhex(base.substring(at + 1));
            if (hash.length < 8 || parent.length < 8) {
                return null;
            }
            return new Ref(hash, parent, file, Files.size(file));
        } catch (NumberFormatException | IOException e) {
            return null;
        }
    }

    /** Entries held: the durable index plus the writes still in flight. */
    int size() {
        return byHash.size() + pending.size();
    }

    /** Whether the tier holds nothing, in flight or on the disk. */
    boolean isEmpty() {
        return byHash.isEmpty() && pending.isEmpty() && loaded.isEmpty();
    }

    /** Bytes the indexed files hold. */
    long bytes() {
        return bytes;
    }

    /** Whether the tier holds the block hashed {@code hash}, durable or in flight. */
    boolean holds(byte[] hash) {
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(hash);
        return byHash.containsKey(k) || pending.containsKey(k) || loaded.containsKey(k);
    }

    /** Whether a file of the block hashed {@code hash} is durable. */
    boolean durable(byte[] hash) {
        return byHash.containsKey(new HostBlockCache.DigestKey(hash));
    }

    /**
     * Accepts {@code entry} for the background writer, replacing the file of the same chain, and never waits for the
     * disk: a full queue, a closed tier, or an entry larger than the whole budget accepts no copy. The block stays
     * promotable from its host image until its file is durable under its final name.
     */
    Spill put(HostBlockCache.Entry entry) {
        if (closed) {
            return Spill.NOT_STORED;
        }
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(entry.hash);
        if (pending.containsKey(k) || loaded.containsKey(k)) {
            // already on its way, or a verified image of the same chain is in memory
            return new Spill(true, Collections.emptyList());
        }
        byte[] blob = blob(entry);
        if (budget >= 0 && blob.length > budget) {
            return Spill.NOT_STORED;
        }
        Path file = dir.resolve(group(entry.parentHash)).resolve(name(entry.hash, entry.parentHash));
        Pending p = new Pending(entry, blob, file);
        pending.put(k, p);
        pendingChildren.computeIfAbsent(p.parentKey, h -> new ArrayList<>(2)).add(p);
        // the slot is taken before the job is visible to the writer, which releases it when the outcome joins
        // written; counting after the offer would let a fast writer finish the job first
        unfinishedWrites.incrementAndGet();
        if (!writes.offer(Write.store(p))) {
            unfinishedWrites.decrementAndGet();
            pending.remove(k);
            unlinkPending(p);
            return Spill.NOT_STORED;
        }
        return new Spill(true, Collections.emptyList());
    }

    /** Serializes {@code entry}: the format {@link #read} answers. */
    private byte[] blob(HostBlockCache.Entry entry) {
        BlockImage image = entry.image;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (DataOutputStream s = new DataOutputStream(out)) {
            s.writeInt(MAGIC);
            s.writeInt(VERSION);
            int[] identity = layout.identity();
            s.writeInt(identity.length);
            for (int v : identity) {
                s.writeInt(v);
            }
            s.writeUTF(hasher.algo().algoName());
            s.write(noneHash);
            int[] chain = entry.chain();
            s.writeInt(chain.length);
            for (int t : chain) {
                s.writeInt(t);
            }
            writeText(s, entry.key.adapters());
            writeText(s, entry.key.loraName());
            writeText(s, entry.key.salt());
            writeBlock(s, image);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return out.toByteArray();
    }

    private static void writeBlock(DataOutputStream s, BlockImage b) throws IOException {
        s.writeInt(b.rows());
        s.writeBoolean(b.full());
        for (int t : b.tokens()) {
            s.writeInt(t);
        }
        for (int p : b.positions()) {
            s.writeInt(p);
        }
        for (int c = 0; c < b.layers(); c++) {
            s.write(b.keys(c));
            writeSideArrays(s, b.kSides(c));
            if (b.values(c) != null) {
                s.write(b.values(c));
                writeSideArrays(s, b.vSides(c));
            }
        }
    }

    private static void writeSideArrays(DataOutputStream s, byte[][] sides) throws IOException {
        for (byte[] side : sides) {
            s.write(side);
        }
    }

    // ------------------------------------------------------------------ reads

    /**
     * The whole block after the block hashed {@code parentHash} under {@code key} whose chain holds
     * {@code want[0, start + pageSize)}: read off the disk, verified and left in place (the caller moves it with
     * {@link #remove}); null when none does.
     */
    HostBlockCache.Entry whole(byte[] parentHash, PrefixKey key, int[] want, int start, int pageSize) {
        HostBlockCache.Entry hit = wholeLoaded(parentHash, key, want, start, pageSize);
        if (hit != null) {
            return hit;
        }
        for (Ref r : new ArrayList<>(childrenOf(parentHash))) {
            if (available(r.hash)) {
                continue;
            }
            HostBlockCache.Entry e = read(r);
            if (e != null && e.image.full() && e.start() == start && e.key.equals(key)
                    && e.continues(want, start + pageSize) == pageSize) {
                touch(r);
                return e;
            }
        }
        return null;
    }

    /**
     * The block after the block hashed {@code parentHash} under {@code key} sharing the longest row run with
     * {@code want[start, limit)} behind the same chain prefix, the last read winning ties; null when none shares
     * a row.
     */
    HostBlockCache.Hit best(byte[] parentHash, PrefixKey key, int[] want, int start, int limit) {
        HostBlockCache.Hit best = bestLoaded(parentHash, key, want, start, limit);
        for (Ref r : new ArrayList<>(childrenOf(parentHash))) {
            if (available(r.hash)) {
                continue;
            }
            HostBlockCache.Entry e = read(r);
            if (e == null || e.start() != start || !e.key.equals(key)) {
                continue;
            }
            int n = e.continues(want, limit);
            if (n > 0 && (best == null || n >= best.rows)) {
                best = new HostBlockCache.Hit(e, n);
            }
        }
        return best;
    }

    /**
     * The entries under {@code key} whose whole chain is a prefix of {@code want}: the disk blocks a history of
     * those tokens left.
     */
    List<HostBlockCache.Entry> within(int[] want, PrefixKey key) {
        List<HostBlockCache.Entry> out = new ArrayList<>();
        Set<HostBlockCache.DigestKey> seen = new HashSet<>();
        for (HostBlockCache.Entry e : loaded.values()) {
            if (within(e, want, key) && seen.add(new HostBlockCache.DigestKey(e.hash))) {
                out.add(e);
            }
        }
        for (Pending p : pending.values()) {
            if (within(p.entry, want, key) && seen.add(p.key)) {
                out.add(p.entry);
            }
        }
        for (Ref r : new ArrayList<>(lru)) {
            if (seen.contains(new HostBlockCache.DigestKey(r.hash))) {
                continue;
            }
            HostBlockCache.Entry e = read(r);
            if (e != null && within(e, want, key)) {
                out.add(e);
            }
        }
        return out;
    }

    private static boolean within(HostBlockCache.Entry e, int[] want, PrefixKey key) {
        return e.key.equals(key) && e.chain().length <= want.length
                && e.continues(want, e.chain().length) == e.image.rows();
    }

    /**
     * The whole block after the block hashed {@code parentHash} under {@code key} whose chain holds
     * {@code want[0, start + pageSize)}, from the images already in memory: a block whose write is still in flight
     * or one a prefetch loaded. Never touches the disk.
     */
    HostBlockCache.Entry wholeLoaded(byte[] parentHash, PrefixKey key, int[] want, int start, int pageSize) {
        for (HostBlockCache.Entry e : loadedChildrenOf(parentHash)) {
            if (e.image.full() && e.start() == start && e.key.equals(key)
                    && e.continues(want, start + pageSize) == pageSize) {
                touchLoaded(e);
                Ref r = byHash.get(new HostBlockCache.DigestKey(e.hash));
                if (r != null) {
                    touch(r);
                }
                return e;
            }
        }
        return null;
    }

    /**
     * The block after the block hashed {@code parentHash} under {@code key} sharing the longest row run with
     * {@code want[start, limit)}, from the images already in memory; never touches the disk.
     */
    HostBlockCache.Hit bestLoaded(byte[] parentHash, PrefixKey key, int[] want, int start, int limit) {
        HostBlockCache.Hit best = null;
        for (HostBlockCache.Entry e : loadedChildrenOf(parentHash)) {
            if (e.start() != start || !e.key.equals(key)) {
                continue;
            }
            int n = e.continues(want, limit);
            if (n > 0 && (best == null || n >= best.rows)) {
                best = new HostBlockCache.Hit(e, n);
            }
        }
        if (best != null) {
            touchLoaded(best.entry);
        }
        return best;
    }

    /**
     * Starts (or checks) the background read of the blocks a promotion of {@code want[0, limit)} under {@code key}
     * would need. Answers true when every block is already in memory or no further block sits on the disk, false
     * while a read is in flight: the scheduler waits for the request rather than blocking the engine's thread.
     */
    boolean prefetchReady(int[] want, PrefixKey key, int limit, int pageSize,
                          Set<HostBlockCache.DigestKey> served) {
        if (closed || limit <= 0) {
            return true;
        }
        PrefetchKey pk = new PrefetchKey(key, Arrays.copyOf(want, limit));
        Prefetch p = prefetches.get(pk);
        if (p != null) {
            if (!p.done) {
                return false;
            }
            prefetches.remove(pk);
            return true;
        }
        List<ReadJob> jobs = discover(want, key, limit, pageSize, served);
        if (jobs.isEmpty()) {
            return true;
        }
        Prefetch started = new Prefetch();
        prefetches.put(pk, started);
        try {
            readers.execute(() -> loadedBlocks.add(readAll(pk, jobs)));
        } catch (RejectedExecutionException e) {
            prefetches.remove(pk);
            return true; // no reader left: the request falls back to computing the rows
        }
        prunePrefetches();
        return false;
    }

    /** The files a promotion of the chain would read: the whole blocks the index knows, then the boundary's
     *  candidates. */
    private List<ReadJob> discover(int[] want, PrefixKey key, int limit, int pageSize,
                                   Set<HostBlockCache.DigestKey> served) {
        List<ReadJob> jobs = new ArrayList<>();
        CacheRows rows = CacheRows.ofTokens(want);
        byte[] parent = noneHash;
        boolean chain = true;
        int pos = 0;
        while (pos + pageSize <= limit) {
            byte[] h = PrefixCache.chainHash(hasher, pos == 0 ? null : parent, rows, pos, pageSize,
                    pos == 0 ? key : null);
            HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(h);
            if (!available(h) && !served.contains(k)) {
                Ref r = byHash.get(k);
                if (r == null) {
                    chain = false;
                    break;
                }
                jobs.add(new ReadJob(r));
            }
            parent = h;
            pos += pageSize;
        }
        if (chain && pos < limit) {
            for (Ref r : new ArrayList<>(childrenOf(pos == 0 ? noneHash : parent))) {
                if (!available(r.hash)) {
                    jobs.add(new ReadJob(r));
                }
            }
        }
        return jobs;
    }

    /** Whether the block hashed {@code hash} already has an image in memory. */
    private boolean available(byte[] hash) {
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(hash);
        return loaded.containsKey(k) || pending.containsKey(k);
    }

    /** Reads and verifies every job of one prefetch on a background thread. */
    private Load readAll(PrefetchKey key, List<ReadJob> jobs) {
        List<HostBlockCache.Entry> entries = new ArrayList<>(jobs.size());
        List<Ref> broken = new ArrayList<>(0);
        for (ReadJob job : jobs) {
            try {
                byte[] blob = io.read(job.ref.file);
                HostBlockCache.Entry e = entry(job.ref, blob);
                if (e != null) {
                    entries.add(e);
                }
            } catch (IOException | RuntimeException e) {
                broken.add(job.ref);
            }
        }
        return new Load(key, entries, broken);
    }

    /** Drops completed prefetches beyond {@link #MAX_PREFETCHES}, oldest completed first. */
    private void prunePrefetches() {
        if (prefetches.size() <= MAX_PREFETCHES) {
            return;
        }
        for (Iterator<Map.Entry<PrefetchKey, Prefetch>> it = prefetches.entrySet().iterator();
                prefetches.size() > MAX_PREFETCHES && it.hasNext(); ) {
            if (it.next().getValue().done) {
                it.remove();
            }
        }
    }

    /** Reads and verifies {@code r} on the caller's thread; a broken file is deleted and a foreign one skipped. */
    private HostBlockCache.Entry read(Ref r) {
        byte[] blob;
        try {
            blob = io.read(r.file);
        } catch (IOException e) {
            LOGGER.warn("disk tier cannot read {}: {}", r.file, e.toString());
            dropBroken(r);
            return null;
        }
        try {
            HostBlockCache.Entry e = entry(r, blob);
            if (e == null) {
                LOGGER.debug("disk tier skips another layout's block file {}", r.file);
            } else {
                addLoaded(e);
            }
            return e;
        } catch (IOException | RuntimeException e) {
            LOGGER.warn("disk tier deletes unreadable block file {}: {}", r.file, e.toString());
            dropBroken(r);
            return null;
        }
    }

    /**
     * The entry {@code blob} holds, checked against {@code r}'s digests, the layout and the hasher: null for a file
     * of another layout or hasher (left alone), an exception for a truncated or corrupted one — a file whose own
     * chain does not hash to its name.
     */
    private HostBlockCache.Entry entry(Ref r, byte[] blob) throws IOException {
        try (DataInputStream s = new DataInputStream(new ByteArrayInputStream(blob))) {
            if (s.readInt() != MAGIC || s.readInt() != VERSION) {
                return null;
            }
            if (!identity(s) || !hasherIdentity(s)) {
                return null;
            }
            int chainLen = s.readInt();
            if (chainLen <= 0 || chainLen > MAX_CHAIN) {
                throw new IOException(r.file + " holds a chain of " + chainLen + " rows");
            }
            int[] chain = new int[chainLen];
            for (int i = 0; i < chainLen; i++) {
                chain[i] = s.readInt();
                if (chain[i] < 0) {
                    throw new IOException(r.file + " holds token " + chain[i] + " in its chain");
                }
            }
            PrefixKey key = PrefixKey.restored(readText(s), readText(s), readText(s));
            BlockImage image = readBlock(s, r);
            int start = chainLen - image.rows();
            if (start < 0) {
                throw new IOException(r.file + " holds a chain of " + chainLen + " rows ending in "
                        + image.rows());
            }
            for (int i = 0; i < image.rows(); i++) {
                if (chain[start + i] != image.tokens()[i]) {
                    throw new IOException(r.file + " holds a chain its block's rows leave");
                }
            }
            boolean root = Arrays.equals(r.parentHash, noneHash);
            byte[] hash = PrefixCache.chainHash(hasher, root ? null : r.parentHash,
                    CacheRows.ofTokens(image.tokens()), 0, image.rows(), root ? key : null);
            if (!Arrays.equals(hash, r.hash)) {
                throw new IOException(r.file + " holds the rows of block " + hex(hash)
                        + ", not its name's " + hex(r.hash));
            }
            return new HostBlockCache.Entry(r.hash, r.parentHash, key, chain, image);
        }
    }

    /** Reads and answers the layout identity, or null for another layout's file. */
    private boolean identity(DataInputStream s) throws IOException {
        int[] want = layout.identity();
        int n = s.readInt();
        if (n != want.length) {
            return false;
        }
        for (int v : want) {
            if (s.readInt() != v) {
                return false;
            }
        }
        return true;
    }

    /** Reads and answers the hasher identity (algorithm name and {@code NONE_HASH}), or null for another's. */
    private boolean hasherIdentity(DataInputStream s) throws IOException {
        if (!hasher.algo().algoName().equals(s.readUTF())) {
            return false;
        }
        byte[] seed = new byte[noneHash.length];
        s.readFully(seed);
        return Arrays.equals(seed, noneHash);
    }

    private BlockImage readBlock(DataInputStream s, Ref r) throws IOException {
        int rows = s.readInt();
        boolean full = s.readBoolean();
        if (full ? rows != pageSize : rows < 1 || rows >= pageSize) {
            throw new IOException(r.file + " holds " + (full ? "whole" : "partial") + " block of " + rows
                    + " rows for pages of " + pageSize);
        }
        int[] tokens = new int[rows];
        for (int i = 0; i < rows; i++) {
            tokens[i] = s.readInt();
            if (tokens[i] < 0) {
                throw new IOException(r.file + " holds token " + tokens[i] + " in its block");
            }
        }
        int[] positions = new int[rows];
        for (int i = 0; i < rows; i++) {
            positions[i] = s.readInt();
            if (positions[i] < 0 || (i > 0 && positions[i] < positions[i - 1])) {
                throw new IOException(r.file + " holds position " + positions[i] + " at row " + i);
            }
        }
        List<StateTransfer.Cache> caches = layout.caches();
        byte[][] k = new byte[caches.size()][];
        byte[][] v = new byte[caches.size()][];
        byte[][][] kSides = new byte[caches.size()][][];
        byte[][][] vSides = new byte[caches.size()][][];
        for (int c = 0; c < caches.size(); c++) {
            StateTransfer.Cache cache = caches.get(c);
            KvLayout l = cache.layout();
            k[c] = new byte[Math.toIntExact(rows * l.rowBytesK())];
            s.readFully(k[c]);
            kSides[c] = readSideArrays(s, cache, false, rows);
            if (!l.vAliasesK()) {
                v[c] = new byte[Math.toIntExact(rows * l.rowBytesV())];
                s.readFully(v[c]);
                vSides[c] = readSideArrays(s, cache, true, rows);
            }
        }
        return new BlockImage(tokens, positions, full, k, v, kSides, vSides);
    }

    private static byte[][] readSideArrays(DataInputStream s, StateTransfer.Cache cache, boolean v, int rows)
            throws IOException {
        java.util.List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> planes = cache.sidePlanes(v);
        byte[][] out = new byte[planes.size()][];
        for (int p = 0; p < planes.size(); p++) {
            long rowBytes = planes.get(p).rowBytesFor(v ? cache.layout().widthV() : cache.layout().widthK(),
                    cache.layout().nHeadKv());
            out[p] = new byte[Math.toIntExact(rows * rowBytes)];
            s.readFully(out[p]);
        }
        return out;
    }

    private static String readText(DataInputStream s) throws IOException {
        return s.readBoolean() ? s.readUTF() : null;
    }

    private static void writeText(DataOutputStream s, String text) throws IOException {
        s.writeBoolean(text != null);
        if (text != null) {
            s.writeUTF(text);
        }
    }

    /** Writes {@code blob} to {@code file} through a temp file and an atomic move. */
    private static void writeFile(byte[] blob, Path file) throws IOException {
        Files.createDirectories(file.getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp-" + ProcessHandle.current().pid() + "-"
                + TEMP_FILES.incrementAndGet());
        try {
            try (DataOutputStream s = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(tmp)))) {
                s.write(blob);
            }
            try {
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException e) {
            Files.deleteIfExists(tmp);
            throw e;
        }
    }

    // ------------------------------------------------------------------ removal

    /** Deletes the file of the block hashed {@code hash} before returning, or none. */
    void remove(byte[] hash) {
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(hash);
        Ref r = byHash.get(k);
        if (r != null) {
            unlink(r);
            deleteFile(r.file);
        }
        cancel(k);
        removeLoaded(k);
    }

    /**
     * Drops the block hashed {@code hash} without waiting for the disk: the writer deletes its file, or drops the
     * queued write. A promotion uses it so the engine's step never waits on a delete.
     */
    void removeAsync(byte[] hash) {
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(hash);
        Ref r = byHash.get(k);
        if (r != null) {
            unlink(r);
            scheduleDelete(r.file);
        }
        cancel(k);
        removeLoaded(k);
    }

    /** Cancels the accepted write of {@code k}, if one is still in flight. */
    private void cancel(HostBlockCache.DigestKey k) {
        Pending p = pending.remove(k);
        if (p != null) {
            p.cancelled = true;
            unlinkPending(p);
        }
    }

    /** Deletes every file of this namespace, answering the dropped hashes oldest first. */
    List<byte[]> clear() {
        List<byte[]> all = new ArrayList<>();
        for (Ref r : lru) {
            all.add(r.hash);
        }
        for (Pending p : pending.values()) {
            p.cancelled = true;
        }
        pending.clear();
        pendingChildren.clear();
        loaded.clear();
        loadedChildren.clear();
        loadedLru.clear();
        loadedBytes = 0;
        prefetches.clear();
        lru.clear();
        children.clear();
        byHash.clear();
        bytes = 0;
        awaitInFlightWrites(BLOCKING_DRAIN_MILLIS);
        deleteNamespaceFiles();
        return all;
    }

    /** Deletes the namespace's block files, including those an abandoned write left behind. */
    private void deleteNamespaceFiles() {
        try (DirectoryStream<Path> groups = Files.newDirectoryStream(dir)) {
            for (Path group : groups) {
                if (!Files.isDirectory(group)) {
                    continue;
                }
                try (DirectoryStream<Path> files = Files.newDirectoryStream(group, "*" + SUFFIX)) {
                    for (Path f : files) {
                        deleteFile(f);
                    }
                }
            }
        } catch (IOException e) {
            LOGGER.warn("disk tier cannot clear {}: {}", dir, e.toString());
        }
    }

    /** Waits (bounded) for queued and in-flight writes, deleting the files they produced. */
    private void awaitInFlightWrites(long millis) {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(millis);
        while (unfinishedWrites.get() > 0 && System.nanoTime() < deadline) {
            discardWritten();
            sleepBriefly();
        }
        discardWritten();
    }

    /** Deletes the files of every completed write whose block is no longer pending. */
    private void discardWritten() {
        WriteDone d;
        while ((d = written.poll()) != null) {
            if (d.ok && pending.get(d.pending.key) != d.pending) {
                deleteFile(d.pending.file);
            }
        }
    }

    private static void sleepBriefly() {
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ------------------------------------------------------------------ background work

    /** The writer: stores and deletes files, never touching the engine's index. */
    private void writeLoop() {
        while (true) {
            Write job;
            try {
                job = writes.take();
            } catch (InterruptedException e) {
                if (closed) {
                    finishAbandonedWrites();
                    return;
                }
                continue;
            }
            if (job == Write.STOP) {
                return;
            }
            if (job.delete != null) {
                deleteFile(job.delete);
                continue;
            }
            Pending p = job.pending;
            if (p.cancelled) {
                writeDone(new WriteDone(p, false, null));
                continue;
            }
            IOException failure = null;
            try {
                io.write(p.blob, p.file);
            } catch (IOException | RuntimeException e) {
                failure = e instanceof IOException ? (IOException) e : new IOException(e.toString(), e);
            } finally {
                writeDone(new WriteDone(p, failure == null, failure));
            }
        }
    }

    /**
     * Records one finished write: the outcome joins {@link #written} and only then is the drain slot
     * {@code put} took released, so a drain that observes no unfinished write also sees every outcome.
     */
    private void writeDone(WriteDone done) {
        written.add(done);
        unfinishedWrites.decrementAndGet();
    }

    /** Completes every job a shutdown dropped from the queue as cancelled, releasing their drain slots. */
    private void finishAbandonedWrites() {
        Write job;
        while ((job = writes.poll()) != null) {
            if (job != Write.STOP && job.delete == null) {
                writeDone(new WriteDone(job.pending, false, null));
            }
        }
    }

    /**
     * Drains what the background work completed since the last call and applies it on the engine's thread: a
     * durable block is indexed (and the budget's drops answered), a block whose write was cancelled is deleted, and
     * the entries a prefetch loaded join the images promotions read.
     */
    List<Event> pump() {
        List<Event> events = new ArrayList<>(deferred);
        deferred.clear();
        WriteDone done;
        while ((done = written.poll()) != null) {
            events.addAll(applyWrite(done));
        }
        Load load;
        while ((load = loadedBlocks.poll()) != null) {
            applyLoad(load);
        }
        return events;
    }

    /** Applies one completed write: indexes a durable block, or deletes a cancelled one. */
    private List<Event> applyWrite(WriteDone done) {
        List<Event> events = new ArrayList<>(1);
        Pending p = done.pending;
        if (pending.get(p.key) != p) {
            // the block left the tier while its write ran (a promotion, a clear or a removal)
            if (done.ok) {
                scheduleDelete(p.file);
            }
            return events;
        }
        pending.remove(p.key);
        unlinkPending(p);
        if (!done.ok) {
            LOGGER.warn("disk tier keeps no copy of block {}: {}", hex(p.entry.hash),
                    done.failure == null ? "cancelled" : done.failure.toString());
            return events;
        }
        Ref old = byHash.get(p.key);
        if (old != null) {
            unlink(old);
        }
        index(new Ref(p.entry.hash, p.entry.parentHash, p.file, p.bytes));
        addLoaded(p.entry);
        events.add(Event.stored(p.entry));
        for (byte[] h : evictForRoom()) {
            events.add(Event.removed(h));
        }
        return events;
    }

    /** Applies one completed prefetch: keeps its verified entries, drops the files that failed verification. */
    private void applyLoad(Load load) {
        Prefetch p = prefetches.get(load.key);
        if (p != null) {
            p.done = true;
        }
        for (Ref broken : load.broken) {
            if (closed) {
                continue; // shutdown interrupted the read: the file is not known to be broken
            }
            Ref current = byHash.get(new HostBlockCache.DigestKey(broken.hash));
            if (current != null && Arrays.equals(current.hash, broken.hash)) {
                dropBroken(current);
            }
        }
        for (HostBlockCache.Entry e : load.entries) {
            if (byHash.containsKey(new HostBlockCache.DigestKey(e.hash))) {
                addLoaded(e);
            }
        }
    }

    /**
     * Stops the background workers, letting the accepted writes finish for {@code timeoutMillis} and abandoning the
     * rest, so shutdown is deterministic rather than waiting on a slow disk. Completed work stays for the next
     * {@link #pump}.
     */
    void close(long timeoutMillis) {
        if (closed) {
            return;
        }
        closed = true;
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
        while (unfinishedWrites.get() > 0 && System.nanoTime() < deadline) {
            sleepBriefly();
        }
        if (unfinishedWrites.get() > 0) {
            LOGGER.warn("disk tier shutdown abandons the writes still pending after {} ms", timeoutMillis);
            for (Pending p : pending.values()) {
                p.cancelled = true;
            }
            pending.clear();
            pendingChildren.clear();
            // the interrupted writer completes the queued jobs as cancelled, releasing their drain slots
            writer.interrupt();
        }
        if (!writes.offer(Write.STOP)) {
            writer.interrupt();
        }
        readers.shutdownNow();
        try {
            writer.join(Math.max(100L, timeoutMillis));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        try {
            readers.awaitTermination(Math.max(100L, timeoutMillis), TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        deferred.addAll(pump());
    }

    /** {@link #close(long)} with the tier's shutdown budget. */
    void close() {
        close(SHUTDOWN_DRAIN_MILLIS);
    }

    /**
     * Waits at most {@code millis} for the admitted writes to become durable, without indexing them; the caller
     * drains the results with {@link #pump}, which answers every finished write's events once this returned true.
     * True when every write finished: the single counter {@link #unfinishedWrites} read at the end answers
     * exactly, with no gap where a taken-but-not-started write looks drained.
     */
    boolean awaitWrites(long millis) {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(millis);
        while (unfinishedWrites.get() > 0 && System.nanoTime() < deadline) {
            sleepBriefly();
        }
        return unfinishedWrites.get() == 0;
    }

    /** The admitted writes the writer has not finished: queued, taken, or executing. */
    int unfinishedWriteCount() {
        return unfinishedWrites.get();
    }

    // ------------------------------------------------------------------ index

    /** Adds {@code r} to the index, replacing the ref of the same digest. */
    private void index(Ref r) {
        byHash.put(new HostBlockCache.DigestKey(r.hash), r);
        children.computeIfAbsent(r.parentKey, h -> new ArrayList<>(2)).add(r);
        lru.add(r);
        bytes += r.bytes;
    }

    /** Drops files least recently used first while the tier holds more than its budget, without waiting on the
     *  deletes. */
    private List<byte[]> evictForRoom() {
        List<byte[]> evicted = new ArrayList<>();
        while (budget >= 0 && bytes > budget && !lru.isEmpty()) {
            Ref oldest = lru.iterator().next();
            unlink(oldest);
            removeLoaded(new HostBlockCache.DigestKey(oldest.hash));
            scheduleDelete(oldest.file);
            evicted.add(oldest.hash);
        }
        return evicted;
    }

    /** Removes {@code r} from the index (the caller deletes its file). */
    private void unlink(Ref r) {
        if (!lru.remove(r)) {
            return;
        }
        byHash.remove(new HostBlockCache.DigestKey(r.hash), r);
        bytes -= r.bytes;
        List<Ref> siblings = children.get(r.parentKey);
        if (siblings != null) {
            siblings.remove(r);
            if (siblings.isEmpty()) {
                children.remove(r.parentKey);
            }
        }
    }

    /** Removes a pending write from its parent's queue. */
    private void unlinkPending(Pending p) {
        List<Pending> siblings = pendingChildren.get(p.parentKey);
        if (siblings != null) {
            siblings.remove(p);
            if (siblings.isEmpty()) {
                pendingChildren.remove(p.parentKey);
            }
        }
    }

    /** Drops a file that failed verification, before returning. */
    private void dropBroken(Ref r) {
        unlink(r);
        removeLoaded(new HostBlockCache.DigestKey(r.hash));
        deleteFile(r.file);
    }

    private void touch(Ref r) {
        if (lru.remove(r)) {
            lru.add(r);
        }
    }

    /** Keeps a verified image for a later promotion, bounded by {@link #residentBudget}. */
    private void addLoaded(HostBlockCache.Entry e) {
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(e.hash);
        if (loaded.putIfAbsent(k, e) != null) {
            return;
        }
        loadedChildren.computeIfAbsent(new HostBlockCache.DigestKey(e.parentHash), h -> new ArrayList<>(2)).add(e);
        loadedLru.add(k);
        loadedBytes += e.bytes();
        while (loadedBytes > residentBudget && loadedLru.size() > 1) {
            removeLoaded(loadedLru.iterator().next());
        }
    }

    /** Forgets the in-memory image of {@code k}; a durable file stays. */
    private void removeLoaded(HostBlockCache.DigestKey k) {
        HostBlockCache.Entry e = loaded.remove(k);
        if (e == null) {
            return;
        }
        loadedLru.remove(k);
        loadedBytes -= e.bytes();
        HostBlockCache.DigestKey parent = new HostBlockCache.DigestKey(e.parentHash);
        List<HostBlockCache.Entry> siblings = loadedChildren.get(parent);
        if (siblings != null) {
            siblings.remove(e);
            if (siblings.isEmpty()) {
                loadedChildren.remove(parent);
            }
        }
    }

    private void touchLoaded(HostBlockCache.Entry e) {
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(e.hash);
        if (loadedLru.remove(k)) {
            loadedLru.add(k);
        }
    }

    /** The in-memory images after {@code parentHash}: those a prefetch loaded and those whose write is in flight. */
    private List<HostBlockCache.Entry> loadedChildrenOf(byte[] parentHash) {
        HostBlockCache.DigestKey k = new HostBlockCache.DigestKey(parentHash);
        List<HostBlockCache.Entry> out = new ArrayList<>(2);
        List<HostBlockCache.Entry> resident = loadedChildren.get(k);
        if (resident != null) {
            out.addAll(resident);
        }
        List<Pending> queued = pendingChildren.get(k);
        if (queued != null) {
            for (Pending p : queued) {
                out.add(p.entry);
            }
        }
        return out;
    }

    /** The refs under {@code parentHash}, or none. */
    private List<Ref> childrenOf(byte[] parentHash) {
        return children.getOrDefault(new HostBlockCache.DigestKey(parentHash), Collections.emptyList());
    }

    /** Queues a file delete on the writer, or deletes it now when the writer already stopped. */
    private void scheduleDelete(Path file) {
        if (!closed && writes.offer(Write.delete(file))) {
            return;
        }
        deleteFile(file);
    }

    private void deleteFile(Path file) {
        try {
            io.delete(file);
        } catch (IOException e) {
            LOGGER.warn("disk tier cannot delete {}: {}", file, e.toString());
        }
    }

    /** The fan-out directory of a parent digest: its leading 2 bytes, hex. */
    private static String group(byte[] parentHash) {
        StringBuilder sb = new StringBuilder(4);
        for (int i = 0; i < 2 && i < parentHash.length; i++) {
            sb.append(Character.forDigit((parentHash[i] >> 4) & 0xf, 16))
                    .append(Character.forDigit(parentHash[i] & 0xf, 16));
        }
        return sb.toString();
    }

    /** The file name of a block: its digest and its parent's, hex. */
    private static String name(byte[] hash, byte[] parentHash) {
        return hex(hash) + "_" + hex(parentHash) + SUFFIX;
    }

    /** The lowercase hex of {@code bytes}. */
    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(Character.forDigit((b >> 4) & 0xf, 16)).append(Character.forDigit(b & 0xf, 16));
        }
        return sb.toString();
    }

    /** The bytes of an even-length hex string. */
    private static byte[] unhex(String hex) {
        if (hex.length() % 2 != 0) {
            throw new NumberFormatException("a block file name holds whole digest bytes, got " + hex);
        }
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < out.length; i++) {
            int hi = Character.digit(hex.charAt(2 * i), 16);
            int lo = Character.digit(hex.charAt(2 * i + 1), 16);
            if (hi < 0 || lo < 0) {
                throw new NumberFormatException("a block file name holds hex digest bytes, got " + hex);
            }
            out[i] = (byte) ((hi << 4) | lo);
        }
        return out;
    }

    @Override
    public String toString() {
        return "DiskBlockCache[" + lru.size() + " blocks, " + bytes + " bytes of "
                + (budget < 0 ? "unbounded" : budget + "") + " under " + dir + "]";
    }
}
