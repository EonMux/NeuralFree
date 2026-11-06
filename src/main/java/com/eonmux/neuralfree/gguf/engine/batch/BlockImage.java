package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The host copy of one prefix-cache block: its token rows, the position each row stores, whether it is a whole
 * block, and the stored K/V bytes of its rows in every owning cache layer (llama.cpp's {@code state_seq_get_data}
 * cell data, one page per KV group). A slot file carries a slot's chain as these images, and the host-RAM prompt
 * cache keeps evicted blocks as them; {@link #install} writes one back into fresh pages. Immutable once read.
 */
final class BlockImage {

    private final int[] tokens;
    private final int[] positions;
    private final boolean full;
    /** Per cache layer (in {@link StateTransfer#caches()} order), the K bytes of the rows. */
    private final byte[][] k;
    /** Per cache layer, the V bytes of the rows; null for a layer whose V aliases K. */
    private final byte[][] v;
    /** Per cache layer and side plane, the stored K scale bytes of the rows (empty for caches without planes). */
    private final byte[][][] kSides;
    private final byte[][][] vSides;

    BlockImage(int[] tokens, int[] positions, boolean full, byte[][] k, byte[][] v) {
        this(tokens, positions, full, k, v, emptySides(v.length), emptySides(v.length));
    }

    BlockImage(int[] tokens, int[] positions, boolean full, byte[][] k, byte[][] v, byte[][][] kSides,
               byte[][][] vSides) {
        this.tokens = Objects.requireNonNull(tokens, "tokens");
        this.positions = Objects.requireNonNull(positions, "positions");
        this.k = Objects.requireNonNull(k, "k");
        this.v = Objects.requireNonNull(v, "v");
        this.kSides = Objects.requireNonNull(kSides, "kSides");
        this.vSides = Objects.requireNonNull(vSides, "vSides");
        if (tokens.length == 0 || tokens.length != positions.length || k.length != v.length
                || kSides.length != k.length || vSides.length != v.length) {
            throw new IllegalArgumentException("a block image of " + tokens.length + " tokens at " + positions.length
                    + " positions over " + k.length + " K and " + v.length + " V layers");
        }
        this.full = full;
    }

    private static byte[][][] emptySides(int caches) {
        byte[][][] out = new byte[caches][][];
        for (int i = 0; i < caches; i++) {
            out[i] = new byte[0][];
        }
        return out;
    }

    /**
     * Reads the first {@code n} rows of cached block {@code block} from the memory: their K/V bytes in every
     * cache layer and the positions they store; {@code full} marks the image whole (all of a page's rows).
     *
     * @throws UnsupportedOperationException when a row holds no token (a media or prompt-embedding row), which an
     *                                       image does not carry
     * @throws IllegalArgumentException      when {@code n} is outside the block or disagrees with {@code full}
     */
    static BlockImage read(PrefixCache.Block block, int n, boolean full, BlockLayout layout) {
        if (n < 1 || n > block.rows.length() || full != (n == layout.pageSize())) {
            throw new IllegalArgumentException((full ? "a whole" : "a partial") + " image of " + n
                    + " rows of a block of " + block.rows.length());
        }
        int[] tokens = block.rows.tokens(0, n);
        for (int t : tokens) {
            if (t == CacheRows.NO_TOKEN) {
                throw new UnsupportedOperationException("a cached block of media or prompt-embedding rows cannot"
                        + " leave the memory; the slot file and the host-RAM cache carry token rows only");
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
            int row0 = block.pages[cache.group()] * layout.pageSize();
            k[c] = new byte[Math.toIntExact(n * l.rowBytesK())];
            cache.read(false, row0, n, k[c]);
            kSides[c] = readSides(cache, false, row0, n);
            if (!l.vAliasesK()) {
                v[c] = new byte[Math.toIntExact(n * l.rowBytesV())];
                cache.read(true, row0, n, v[c]);
                vSides[c] = readSides(cache, true, row0, n);
            }
        }
        int[] positions = layout.paged().pagePositions(0, block.pages[0], n);
        return new BlockImage(tokens, positions, full, k, v, kSides, vSides);
    }

    /** Reads every side plane of K ({@code v} false) or V of {@code n} rows into one byte array per plane. */
    static byte[][] readSides(StateTransfer.Cache cache, boolean v, int row0, int n) {
        List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> planes = cache.sidePlanes(v);
        byte[][] out = new byte[planes.size()][];
        for (int p = 0; p < planes.size(); p++) {
            long rowBytes = planes.get(p).rowBytesFor(v ? cache.layout().widthV() : cache.layout().widthK(),
                    cache.layout().nHeadKv());
            out[p] = new byte[Math.toIntExact(n * rowBytes)];
            cache.readSide(v, p, row0, n, out[p]);
        }
        return out;
    }

    /**
     * Writes the image's rows into {@code pages} (one per KV group, taken by
     * {@link com.eonmux.neuralfree.gguf.memory.paged.PagedSequences#importPages}).
     *
     * @throws IllegalArgumentException when the image was read for other cache layers than {@code layout}'s
     */
    void install(BlockLayout layout, int[] pages) {
        List<StateTransfer.Cache> caches = layout.caches();
        if (caches.size() != k.length) {
            throw new IllegalArgumentException("a block image of " + k.length + " cache layers cannot fill a memory of "
                    + caches.size());
        }
        int n = tokens.length;
        for (int c = 0; c < caches.size(); c++) {
            StateTransfer.Cache cache = caches.get(c);
            int row0 = pages[cache.group()] * layout.pageSize();
            cache.write(false, row0, n, k[c]);
            writeSides(cache, false, row0, n, kSides[c]);
            if (!cache.layout().vAliasesK()) {
                cache.write(true, row0, n, v[c]);
                writeSides(cache, true, row0, n, vSides[c]);
            }
        }
    }

    /** Writes every side plane of K ({@code v} false) or V of {@code n} rows. */
    static void writeSides(StateTransfer.Cache cache, boolean v, int row0, int n, byte[][] sides) {
        List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> planes = cache.sidePlanes(v);
        if (planes.size() != sides.length) {
            throw new IllegalArgumentException("the block image holds " + sides.length + " side planes, the cache "
                    + planes.size());
        }
        for (int p = 0; p < planes.size(); p++) {
            cache.writeSide(v, p, row0, n, sides[p]);
        }
    }

    /**
     * The partial image of this image's first {@code n} rows, for a chain that shares only those.
     *
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, rows())} or the layout's caches differ
     */
    BlockImage head(int n, BlockLayout layout) {
        List<StateTransfer.Cache> caches = layout.caches();
        if (n < 1 || n >= tokens.length || caches.size() != k.length) {
            throw new IllegalArgumentException("the head of " + n + " rows of a block image of " + tokens.length
                    + " rows over " + k.length + " cache layers");
        }
        byte[][] hk = new byte[k.length][];
        byte[][] hv = new byte[k.length][];
        byte[][][] hks = new byte[k.length][][];
        byte[][][] hvs = new byte[k.length][][];
        for (int c = 0; c < k.length; c++) {
            StateTransfer.Cache cache = caches.get(c);
            KvLayout l = cache.layout();
            hk[c] = Arrays.copyOf(k[c], Math.toIntExact(n * l.rowBytesK()));
            hv[c] = v[c] == null ? null : Arrays.copyOf(v[c], Math.toIntExact(n * l.rowBytesV()));
            hks[c] = headSides(cache, false, n, kSides[c]);
            hvs[c] = vSides[c] == null ? null : headSides(cache, true, n, vSides[c]);
        }
        return new BlockImage(Arrays.copyOf(tokens, n), Arrays.copyOf(positions, n), false, hk, hv, hks, hvs);
    }

    /** The first {@code n} rows of every side plane of a cache. */
    private static byte[][] headSides(StateTransfer.Cache cache, boolean v, int n, byte[][] sides) {
        List<com.eonmux.neuralfree.gguf.memory.KvDType.SidePlane> planes = cache.sidePlanes(v);
        byte[][] out = new byte[planes.size()][];
        for (int p = 0; p < planes.size(); p++) {
            int rowBytes = Math.toIntExact(planes.get(p).rowBytesFor(
                    v ? cache.layout().widthV() : cache.layout().widthK(), cache.layout().nHeadKv()));
            out[p] = Arrays.copyOf(sides[p], n * rowBytes);
        }
        return out;
    }

    /** The stored K side-plane bytes of cache layer {@code c} (one array per plane; empty without planes). */
    byte[][] kSides(int c) {
        return kSides[c];
    }

    /** The stored V side-plane bytes of cache layer {@code c}; null for a layer whose V aliases K. */
    byte[][] vSides(int c) {
        return vSides[c];
    }

    /** Rows in the block. */
    int rows() {
        return tokens.length;
    }

    /** Whether the block is whole (a page of rows); a partial block ends its chain. */
    boolean full() {
        return full;
    }

    /** The token of every row (a copy). */
    int[] tokens() {
        return tokens.clone();
    }

    /** The position every row stores (a copy). */
    int[] positions() {
        return positions.clone();
    }

    /** Cache layers the image covers. */
    int layers() {
        return k.length;
    }

    /** The K bytes of cache layer {@code c} (the live array; callers only read it). */
    byte[] keys(int c) {
        return k[c];
    }

    /** The V bytes of cache layer {@code c} (the live array; callers only read it), null when V aliases K. */
    byte[] values(int c) {
        return v[c];
    }

    /** Host bytes the image holds: its K/V rows, their side planes, tokens and positions. */
    long bytes() {
        long total = 8L * tokens.length;
        for (int c = 0; c < k.length; c++) {
            total += k[c].length + (v[c] == null ? 0 : v[c].length) + sidesBytes(kSides[c])
                    + (vSides[c] == null ? 0 : sidesBytes(vSides[c]));
        }
        return total;
    }

    private static long sidesBytes(byte[][] sides) {
        long total = 0;
        for (byte[] side : sides) {
            total += side.length;
        }
        return total;
    }
}
