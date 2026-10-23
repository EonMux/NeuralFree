package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;

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
     * stored in each row, -1 when empty) and one per-step {@code cell_idx} buffer.
     */
    public static final class KvGroup {
        private final int index;
        private final int rows;
        private final boolean ring;
        private final KvWindow window;

        private KvGroup(int index, int rows, boolean ring, KvWindow window) {
            this.index = index;
            this.rows = rows;
            this.ring = ring;
            this.window = window;
        }

        /** @throws IllegalArgumentException for a negative index or non-positive rows */
        public static KvGroup of(int index, int rows, boolean ring, KvWindow window) {
            if (index < 0 || rows <= 0) {
                throw new IllegalArgumentException("KV group needs index >= 0 and rows > 0, got " + index + "/" + rows);
            }
            return new KvGroup(index, rows, ring, Objects.requireNonNull(window, "window"));
        }

        public int index() { return index; }
        public int rows() { return rows; }
        public boolean ring() { return ring; }
        public KvWindow window() { return window; }

        /** Row of a position in this group ({@code pos % rows} for a ring). */
        public int rowOf(int pos) {
            if (pos < 0) {
                throw new IllegalArgumentException("KV position " + pos + " is negative");
            }
            if (!ring && pos >= rows) {
                throw new IllegalArgumentException("KV position " + pos + " is beyond the group of " + rows + " rows");
            }
            return ring ? pos % rows : pos;
        }

        /** Bytes of the {@code kv_pos} buffer. */
        public long kvPosBytes() {
            return (long) rows * Integer.BYTES;
        }

        @Override
        public String toString() {
            return "KvGroup{" + index + ", rows=" + rows + (ring ? " ring" : "") + ", " + window + '}';
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
     * most {@code maxStep}: {@code min(maxStep, remaining)}, except that with {@link #rollbackSnapshots()}
     * {@code s > 0} the trailing {@code 1 + s} tokens of the call stay in one step, as llama.cpp splits a batch for
     * memory with rollback snapshots ({@code split_equal} with {@code n_keep_tail = 1 + n_rs_seq},
     * src/llama-batch.cpp:714-765, src/llama-kv-cache-dsv4.cpp:1414-1417): a full step that would leave between 1
     * and {@code s} tokens stops early so exactly {@code 1 + s} remain. The snapshots are written from the last step
     * only, so this keeps a rollback of up to {@code s} positions after the call possible. A step of at most
     * {@code 1 + s} tokens cannot keep the tail apart and splits plainly (llama.cpp asserts
     * {@code n_ubatch > n_keep_tail}).
     *
     * @throws IllegalArgumentException for {@code remaining < 1} or {@code maxStep < 1}
     */
    public int stepTokens(int remaining, int maxStep) {
        if (remaining < 1 || maxStep < 1) {
            throw new IllegalArgumentException("a step needs remaining >= 1 and maxStep >= 1, got " + remaining + "/"
                    + maxStep);
        }
        int n = Math.min(maxStep, remaining);
        int keepTail = rollbackSnapshots() > 0 ? 1 + rollbackSnapshots() : 0;
        int left = remaining - n;
        if (keepTail == 0 || left == 0 || left >= keepTail || maxStep <= keepTail) {
            return n;
        }
        return remaining - keepTail;
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

    @Override
    public String toString() {
        return "MemoryPlan{" + kind + ", nCtx=" + nCtx + ", ubatch=" + ubatch + ", layers=" + layers.size() + ", groups="
                + groups.size() + ", kv=" + kvBytes() + "B, state=" + stateBytes() + "B, snapshots=" + snapshotBytes()
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
