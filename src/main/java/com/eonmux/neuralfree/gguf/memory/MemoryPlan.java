package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.engine.StepSizes;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Complete memory plan of a model for one context size (built by {@code MemoryPlanner}): per-layer plans indexed by
 * virtual layer ({@code il}; trunk then NextN layers), the KV groups that share one {@code kv_pos[rows]} position
 * buffer, the settings it was planned with, and byte totals. Immutable.
 */
public final class MemoryPlan {

    /**
     * A set of layers whose caches share row addressing: one device {@code int kv_pos[rows]} buffer (the position
     * stored in each row, -1 when empty) and one per-step {@code cell_idx} buffer. Rows of a paged group live in the
     * {@code pages} pages of {@link #pageSize()} rows each, and {@code PagedSequences} maps positions to rows;
     * {@link #rowOf(int)} addresses the rows of a contiguous group ({@link #pageSize()} 0) only.
     */
    public static final class KvGroup {
        private final int index;
        private final int rows;
        private final boolean ring;
        private final KvWindow window;
        private final int pageSize;
        private final int pages;
        private final int rowBase;
        private final boolean sharded;

        private KvGroup(int index, int rows, boolean ring, KvWindow window, int pageSize, int pages) {
            this(index, rows, ring, window, pageSize, pages, 0, false);
        }

        private KvGroup(int index, int rows, boolean ring, KvWindow window, int pageSize, int pages, int rowBase,
                        boolean sharded) {
            this.index = index;
            this.rows = rows;
            this.ring = ring;
            this.window = window;
            this.pageSize = pageSize;
            this.pages = pages;
            this.rowBase = rowBase;
            this.sharded = sharded;
        }

        /** @throws IllegalArgumentException for a negative index or non-positive rows */
        public static KvGroup of(int index, int rows, boolean ring, KvWindow window) {
            if (index < 0 || rows <= 0) {
                throw new IllegalArgumentException("KV group needs index >= 0 and rows > 0, got " + index + "/" + rows);
            }
            return new KvGroup(index, rows, ring, Objects.requireNonNull(window, "window"), 0, 0);
        }

        /**
         * Paged form: {@code rows == pages x pageSize} rows in {@code pages} pages of a page pool, {@code pageSize}
         * a power of two in {@code [1, 256]}. The paged form never wraps ({@code ring} false).
         *
         * @throws IllegalArgumentException for a negative index, non-positive rows, a page size that is not a power
         *                                  of two in {@code [1, 256]}, rows that are not {@code pages x pageSize},
         *                                  or a ring
         */
        public static KvGroup of(int index, int rows, boolean ring, KvWindow window, int pageSize, int pages) {
            if (index < 0 || rows <= 0) {
                throw new IllegalArgumentException("KV group needs index >= 0 and rows > 0, got " + index + "/" + rows);
            }
            if (pageSize > 0) {
                if (pageSize < 1 || pageSize > MemoryConfig.MAX_PAGE_SIZE || Integer.bitCount(pageSize) != 1) {
                    throw new IllegalArgumentException("KV group page size must be a power of two in [1, "
                            + MemoryConfig.MAX_PAGE_SIZE + "], got " + pageSize);
                }
                if (pages <= 0) {
                    throw new IllegalArgumentException("paged KV group needs pages > 0, got " + pages);
                }
                if ((long) pages * pageSize != rows) {
                    throw new IllegalArgumentException("paged KV group rows " + rows + " are not " + pages
                            + " pages of " + pageSize);
                }
                if (ring) {
                    throw new IllegalArgumentException("paged KV groups never wrap (ring)");
                }
                return new KvGroup(index, rows, false, Objects.requireNonNull(window, "window"), pageSize, pages);
            }
            if (pages != 0) {
                throw new IllegalArgumentException("KV group needs pages only with a positive page size, got " + pages);
            }
            return new KvGroup(index, rows, ring, Objects.requireNonNull(window, "window"), 0, 0);
        }

        /**
         * Context-shard form: rows {@code [rowBase, rowBase + rows)} of the context, the block one rank of a
         * context-parallel split keeps (vLLM's decode context parallelism). Never paged and never a ring.
         *
         * @throws IllegalArgumentException for a negative index or base, or non-positive rows
         */
        public static KvGroup ofShard(int index, int rows, int rowBase, KvWindow window) {
            if (index < 0 || rows <= 0) {
                throw new IllegalArgumentException("KV group needs index >= 0 and rows > 0, got " + index + "/" + rows);
            }
            if (rowBase < 0) {
                throw new IllegalArgumentException("KV group needs a non-negative row base, got " + rowBase);
            }
            return new KvGroup(index, rows, false, Objects.requireNonNull(window, "window"), 0, 0, rowBase, true);
        }

        public int index() { return index; }
        public int rows() { return rows; }
        public boolean ring() { return ring; }
        public KvWindow window() { return window; }
        /** Page size P of the paged group's page pool; 0 for the contiguous form. */
        public int pageSize() { return pageSize; }
        /** Pages of the paged group's page pool; 0 for the contiguous form. */
        public int pages() { return pages; }
        /** True when the group's rows live in the pages of a page pool ({@link #pageSize()} over 0). */
        public boolean paged() { return pageSize > 0; }

        /** Global position of row 0 of a context shard; 0 for a whole cache. */
        public int rowBase() { return rowBase; }

        /** True when the group is one rank's context shard, covering positions {@code [rowBase, rowBase + rows)}. */
        public boolean sharded() { return sharded; }

        /** Row of a position in this group ({@code pos % rows} for a ring, {@code pos - rowBase} for a shard). */
        public int rowOf(int pos) {
            if (paged()) {
                throw new UnsupportedOperationException("a paged KV group maps positions through PagedSequences; "
                        + "rowOf is used only for a contiguous group (pageSize 0)");
            }
            if (pos < 0) {
                throw new IllegalArgumentException("KV position " + pos + " is negative");
            }
            if (sharded) {
                if (pos < rowBase || pos >= rowBase + rows) {
                    throw new IllegalArgumentException("KV position " + pos + " is outside the context shard of rows ["
                            + rowBase + ", " + (rowBase + rows) + ")");
                }
                return pos - rowBase;
            }
            if (!ring && pos >= rows) {
                throw new IllegalArgumentException("KV position " + pos + " is beyond the group of " + rows + " rows");
            }
            return ring ? pos % rows : pos;
        }

        /** True when the group, or the context shard, stores position {@code pos}. */
        public boolean holds(int pos) {
            if (pos < 0) {
                return false;
            }
            if (paged()) {
                return pos < rows;
            }
            return ring || (sharded ? pos >= rowBase && pos < rowBase + rows : pos < rows);
        }

        /** Bytes of the {@code kv_pos} buffer. */
        public long kvPosBytes() {
            return (long) rows * Integer.BYTES;
        }

        @Override
        public String toString() {
            return "KvGroup{" + index + ", rows=" + rows + (ring ? " ring" : "")
                    + (sharded ? " shard@" + rowBase : "")
                    + (paged() ? ", pages=" + pages + "x" + pageSize : "") + ", " + window + '}';
        }
    }

    private final MemoryKind kind;
    private final int nCtx;
    private final int ubatch;
    private final List<LayerMemoryPlan> layers;
    private final List<KvGroup> groups;
    private final MemoryConfig config;

    private MemoryPlan(Builder b) {
        this.kind = b.kind;
        this.nCtx = b.nCtx;
        this.ubatch = b.ubatch;
        this.layers = Collections.unmodifiableList(new ArrayList<>(b.layers));
        this.groups = Collections.unmodifiableList(new ArrayList<>(b.groups));
        this.config = b.config;
    }

    public static Builder builder(MemoryKind kind, int nCtx, int ubatch) {
        return new Builder(kind, nCtx, ubatch);
    }

    public MemoryKind kind() { return kind; }
    /** Context length in tokens. */
    public int nCtx() { return nCtx; }
    /** Maximum tokens per forward step. */
    public int ubatch() { return ubatch; }
    /** Per-layer plans; index {@code i} has {@code layer() == i} (unmodifiable). */
    public List<LayerMemoryPlan> layers() { return layers; }
    public LayerMemoryPlan layer(int il) {
        if (il < 0 || il >= layers.size()) {
            throw new IndexOutOfBoundsException("memory plan has no layer " + il + " (" + layers.size() + " layers)");
        }
        return layers.get(il);
    }
    /** KV groups; index {@code g} has {@code index() == g} (unmodifiable). */
    public List<KvGroup> groups() { return groups; }
    /** Pages of the largest paged group's pool; the page ids of every group stay below it. */
    public int maxGroupPages() {
        int max = 0;
        for (KvGroup g : groups) {
            max = Math.max(max, g.pages());
        }
        return max;
    }
    public KvGroup group(int g) {
        if (g < 0 || g >= groups.size()) {
            throw new IndexOutOfBoundsException("memory plan has no KV group " + g + " (" + groups.size() + " groups)");
        }
        return groups.get(g);
    }

    /**
     * The settings the plan was built with ({@code MemoryPlanner} always records them); null for a plan assembled
     * directly with the {@link Builder} without {@link Builder#config(MemoryConfig)}.
     */
    public MemoryConfig config() { return config; }

    /** True when any layer keeps recurrent state. */
    public boolean hasRecurrentState() {
        for (LayerMemoryPlan l : layers) {
            if (l.hasState()) {
                return true;
            }
        }
        return false;
    }

    /**
     * True when the plan keeps recurrent state and every state tensor is
     * {@link LayerMemoryPlan.StateLayout#positional() positional}: the positions it holds are re-evaluated after a
     * drop, so dropping trailing positions leaves no state to restore ({@link ModelMemory#dropRecomputed(int)}).
     */
    public boolean hasOnlyPositionalState() {
        boolean any = false;
        for (LayerMemoryPlan l : layers) {
            for (LayerMemoryPlan.StateTensor s : l.states()) {
                any = true;
                if (!s.layout().positional()) {
                    return false;
                }
            }
        }
        return any;
    }

    /** True when any layer keeps a {@link LayerMemoryPlan.StateLayout#positional() positional} state tensor. */
    public boolean hasPositionalState() {
        for (LayerMemoryPlan l : layers) {
            for (LayerMemoryPlan.StateTensor s : l.states()) {
                if (s.layout().positional()) {
                    return true;
                }
            }
        }
        return false;
    }

    /** True when any KV group is a ring. */
    public boolean hasRing() {
        for (KvGroup g : groups) {
            if (g.ring()) {
                return true;
            }
        }
        return false;
    }

    /** True when any layer has DSV4 compressed caches, whose compressor state accumulates across positions. */
    public boolean hasCompressed() {
        for (LayerMemoryPlan l : layers) {
            if (!l.compressed().isEmpty()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Rollback snapshots of the DSV4 compressor state ({@code n_rs_seq}): the most trailing positions one rollback
     * can drop from memory with compressed caches; 0 when no layer has compressed caches or none keeps snapshots.
     */
    public int rollbackSnapshots() {
        int n = 0;
        for (LayerMemoryPlan l : layers) {
            for (LayerMemoryPlan.CompressedKvPlan c : l.compressed()) {
                n = Math.max(n, c.snapshots());
            }
        }
        return n;
    }

    /**
     * Tokens of the next step when {@code remaining} tokens of one call are still to be stored and a step holds at
     * most {@code maxStep}: the call splits into as few steps as {@code maxStep} allows, of equal size up to one
     * token ({@link StepSizes}), so a call of 19014 tokens at a cap of 8192 runs as 6338, 6338 and 6338 instead of
     * 8192, 8192 and 2630 — a short step costs nearly as much as a full one when a step pays a fixed cost.
     *
     * <p>With {@link #rollbackSnapshots()} {@code s > 0} the trailing {@code 1 + s} tokens of the call stay in the
     * last step, as llama.cpp splits a batch for memory with rollback snapshots ({@code split_equal} with
     * {@code n_keep_tail = 1 + n_rs_seq}, src/llama-batch.cpp:714-765,
     * src/llama-kv-cache-dsv4.cpp:1414-1417): when an equal split would leave fewer than {@code 1 + s} tokens
     * there, the steps before it give up the tokens it needs. The snapshots are written from the last step only, so
     * this keeps a rollback of up to {@code s} positions after the call possible. A step of at most {@code 1 + s}
     * tokens cannot keep the tail apart and splits plainly (llama.cpp asserts {@code n_ubatch > n_keep_tail}).
     *
     * @throws IllegalArgumentException for {@code remaining < 1} or {@code maxStep < 1}
     */
    public int stepTokens(int remaining, int maxStep) {
        int keepTail = rollbackSnapshots() > 0 ? 1 + rollbackSnapshots() : 0;
        return StepSizes.next(remaining, maxStep, keepTail);
    }

    /**
     * True when the memory keeps a shadow copy of every recurrent state for {@code snapshot()} / {@code restore()}:
     * the settings ask for it ({@link MemoryConfig#stateSnapshots()}), some layer keeps recurrent state and no layer
     * has DSV4 compressor state (which a snapshot would miss). The shadows are allocated with the rest of the memory,
     * as llama.cpp allocates its {@code n_rs_seq} rollback rows with the recurrent cache
     * (src/llama-memory-recurrent.cpp:101) when speculative decoding asks for them (common/common.cpp:1720).
     */
    public boolean snapshotsReserved() {
        return config != null && config.stateSnapshots() && hasRecurrentState() && !hasCompressed();
    }

    public long kvBytes() {
        long b = 0;
        for (LayerMemoryPlan l : layers) {
            b += l.kvBytes();
        }
        return b;
    }

    public long stateBytes() {
        long b = 0;
        for (LayerMemoryPlan l : layers) {
            b += l.stateBytes();
        }
        return b;
    }

    public long sideCacheBytes() {
        long b = 0;
        for (LayerMemoryPlan l : layers) {
            b += l.sideCacheBytes();
        }
        return b;
    }

    /** Bytes of the recurrent-state shadows: {@link #stateBytes()} when {@link #snapshotsReserved()}, else 0. */
    public long snapshotBytes() {
        return snapshotsReserved() ? stateBytes() : 0L;
    }

    /** Bytes of the {@code kv_pos} buffers. */
    public long positionBytes() {
        long b = 0;
        for (KvGroup g : groups) {
            b += g.kvPosBytes();
        }
        return b;
    }

    /** Sum of all categories: the layer plans, the recurrent-state shadows and the position buffers. */
    public long totalBytes() {
        return kvBytes() + stateBytes() + snapshotBytes() + sideCacheBytes() + positionBytes();
    }

    /** The plan's kind, sizes, KV groups (rows, ring or page pool, window) and byte totals per category. */
    @Override
    public String toString() {
        return "MemoryPlan{" + kind + ", nCtx=" + nCtx + ", ubatch=" + ubatch + ", layers=" + layers.size() + ", groups="
                + groups + ", kv=" + kvBytes() + "B, state=" + stateBytes() + "B, snapshots=" + snapshotBytes()
                + "B, side=" + sideCacheBytes() + "B, pos=" + positionBytes() + "B}";
    }

    public static final class Builder {
        private final MemoryKind kind;
        private final int nCtx;
        private final int ubatch;
        private final List<LayerMemoryPlan> layers = new ArrayList<>();
        private final List<KvGroup> groups = new ArrayList<>();
        private MemoryConfig config;

        private Builder(MemoryKind kind, int nCtx, int ubatch) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.nCtx = nCtx;
            this.ubatch = ubatch;
        }

        public Builder layer(LayerMemoryPlan v) { this.layers.add(Objects.requireNonNull(v, "layer")); return this; }
        public Builder group(KvGroup v) { this.groups.add(Objects.requireNonNull(v, "group")); return this; }
        /** The settings the plan is built with. */
        public Builder config(MemoryConfig v) { this.config = Objects.requireNonNull(v, "config"); return this; }

        /** @throws IllegalArgumentException when indices are not contiguous or a layer disagrees with its group */
        public MemoryPlan build() {
            if (nCtx <= 0 || ubatch <= 0) {
                throw new IllegalArgumentException("memory plan needs positive nCtx and ubatch, got " + nCtx + "/" + ubatch);
            }
            for (int g = 0; g < groups.size(); g++) {
                if (groups.get(g).index() != g) {
                    throw new IllegalArgumentException("KV group at position " + g + " has index " + groups.get(g).index());
                }
            }
            for (int i = 0; i < layers.size(); i++) {
                LayerMemoryPlan l = layers.get(i);
                if (l.layer() != i) {
                    throw new IllegalArgumentException("memory plan at position " + i + " is for layer " + l.layer());
                }
                if (l.kvGroup() >= 0) {
                    if (l.kvGroup() >= groups.size()) {
                        throw new IllegalArgumentException("layer " + i + " uses KV group " + l.kvGroup() + " of "
                                + groups.size());
                    }
                    KvGroup g = groups.get(l.kvGroup());
                    if (g.rows() != l.kv().rows() || g.ring() != l.kv().ring()) {
                        throw new IllegalArgumentException("layer " + i + " layout " + l.kv() + " disagrees with " + g);
                    }
                }
                if (l.sharedFrom() != null && !l.sharedFrom().target()) {
                    int owner = l.sharedFrom().layer();
                    if (owner >= layers.size() || !layers.get(owner).ownsKv()) {
                        throw new IllegalArgumentException("layer " + i + " shares the KV of layer " + owner
                                + ", which owns no cache");
                    }
                    if (l.kvGroup() >= 0 && l.kvGroup() != layers.get(owner).kvGroup()) {
                        throw new IllegalArgumentException("layer " + i + " shares the KV of layer " + owner
                                + " but addresses it through KV group " + l.kvGroup() + " instead of "
                                + layers.get(owner).kvGroup());
                    }
                }
            }
            return new MemoryPlan(this);
        }
    }
}
