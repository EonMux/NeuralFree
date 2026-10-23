package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Memory of one virtual layer: its KV cache layout (or the layer whose cache it reads), recurrent state tensors,
 * and the sparse-attention side caches (DSA indexer, MSA index keys, DSV4 compressed caches). Byte counts cover
 * only memory this layer owns. Immutable.
 */
public final class LayerMemoryPlan {

    /** What memory the layer owns. */
    public enum Kind {
        /** No memory (FFN-only layers, encoder layers, non-causal passes). */
        NONE,
        /** Own KV cache. */
        KV,
        /** Reads another layer's KV cache ({@link #sharedFrom()}). */
        SHARED_KV,
        /** Recurrent state only. */
        RECURRENT,
        /** KV cache plus recurrent state (falcon-h1 parallel attention, qwen4exp PLE convolution history). */
        KV_RECURRENT,
        /** DeepSeek-V4: raw window cache plus compressed caches and compressor states. */
        DSV4
    }

    /**
     * Element order of a recurrent state tensor (the llama.cpp layouts; fastest index last). All
     * recurrent states are F32 and hold one row per sequence ({@code n_seqs = 1}).
     */
    public enum StateLayout {
        /** {@code conv[c][j]}, {@code j in [0, d_conv - 1)}, {@code j = 0} oldest (mamba1/2, plamo2, GDN, shortconv). */
        CONV,
        /** {@code [q | k | v]} slices, each {@code [c][j]} as {@link #CONV} (KDA). */
        KDA_CONV,
        /** {@code s[(h * head_dim + p) * d_state + n]} (mamba scan). */
        MAMBA_SCAN,
        /** {@code s[h * S * S + i * S + j]}, {@code i} = key, {@code j} = value (wkv6 / gla). */
        WKV6,
        /** {@code s[h * S * S + i * S + j]}, {@code i} = value, {@code j} = key (wkv7). */
        WKV7,
        /** {@code s[h * Sv * Sk + col * Sk + i]}, {@code col} = value, {@code i} = key (GDN / KDA). */
        DELTA,
        /** {@code s[h * D * D + v * D + k]} (lightning attention). */
        LIGHTNING,
        /** {@code [att_shift(n_embd) | ffn_shift(n_embd)]}, the second half only with two shift slots (RWKV). */
        TOKEN_SHIFT,
        /** qwen4exp PLE convolution history: {@code (ple_conv_kernel - 1) * ple_ngram_size * hc_mult * n_embd}. */
        PLE_CONV
    }

    /** One F32 recurrent state tensor of a layer. */
    public static final class StateTensor {
        private final String name;
        private final StateLayout layout;
        private final long floats;

        private StateTensor(String name, StateLayout layout, long floats) {
            this.name = name;
            this.layout = layout;
            this.floats = floats;
        }

        /** @throws IllegalArgumentException for an empty name or a non-positive size */
        public static StateTensor of(String name, StateLayout layout, long floats) {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("state tensor name is empty");
            }
            Objects.requireNonNull(layout, "layout");
            if (floats <= 0) {
                throw new IllegalArgumentException("state tensor " + name + " needs a positive size, got " + floats);
            }
            return new StateTensor(name, layout, floats);
        }

        public String name() { return name; }
        public StateLayout layout() { return layout; }
        /** Floats per sequence. */
        public long floats() { return floats; }
        public long bytes() { return floats * Float.BYTES; }

        @Override
        public String toString() {
            return name + "[" + layout + ", " + floats + " f32]";
        }
    }

    /** State tensor names used by {@link #recurrentState(RecurrentSpec)} and the PLE history. */
    public static final String STATE_CONV = "conv";
    public static final String STATE_SCAN = "state";
    public static final String STATE_SHIFT = "shift";
    public static final String STATE_PLE_CONV = "ple_conv";

    /**
     * The recurrent state tensors of a recurrent mixer: the "r" tensor ({@link #STATE_CONV}, or
     * {@link #STATE_SHIFT} for RWKV) and the "s" tensor ({@link #STATE_SCAN}), sized as llama.cpp
     * {@code n_embd_r()} / {@code n_embd_s()}. The centered short convolution of the LFM2 decision layers is stateless
     * (lfm2.cpp:355-381) and has none.
     */
    public static List<StateTensor> recurrentState(RecurrentSpec spec) {
        Objects.requireNonNull(spec, "spec");
        List<StateTensor> out = new ArrayList<>(2);
        MixerKind k = spec.kind();
        long r = spec.convStateFloats();
        switch (k) {
            case RWKV6: case RWKV6_QWEN2:
                out.add(StateTensor.of(STATE_SHIFT, StateLayout.TOKEN_SHIFT, r));
                out.add(StateTensor.of(STATE_SCAN, StateLayout.WKV6, spec.stateFloats()));
                break;
            case RWKV7:
                out.add(StateTensor.of(STATE_SHIFT, StateLayout.TOKEN_SHIFT, r));
                out.add(StateTensor.of(STATE_SCAN, StateLayout.WKV7, spec.stateFloats()));
                break;
            case SHORTCONV:
                out.add(StateTensor.of(STATE_CONV, StateLayout.CONV, r));
                break;
            case SHORTCONV_CENTERED:
                break;
            case LIGHTNING:
                out.add(StateTensor.of(STATE_SCAN, StateLayout.LIGHTNING, spec.stateFloats()));
                break;
            case KDA:
                out.add(StateTensor.of(STATE_CONV, StateLayout.KDA_CONV, r));
                out.add(StateTensor.of(STATE_SCAN, StateLayout.DELTA, spec.stateFloats()));
                break;
            case GDN:
                out.add(StateTensor.of(STATE_CONV, StateLayout.CONV, r));
                out.add(StateTensor.of(STATE_SCAN, StateLayout.DELTA, spec.stateFloats()));
                break;
            default:
                out.add(StateTensor.of(STATE_CONV, StateLayout.CONV, r));
                out.add(StateTensor.of(STATE_SCAN, StateLayout.MAMBA_SCAN, spec.stateFloats()));
                break;
        }
        return Collections.unmodifiableList(out);
    }

    /** Where a shared-KV layer reads its cache: a layer of this model, or of the target model (drafters). */
    public static final class SharedKv {
        private final int layer;
        private final boolean target;

        private SharedKv(int layer, boolean target) {
            this.layer = layer;
            this.target = target;
        }

        /** Owner layer of this model ({@code kvOwnerLayer}). */
        public static SharedKv local(int ownerLayer) {
            return new SharedKv(checkLayer(ownerLayer), false);
        }

        /** Layer of the target model (gemma4-assistant reads target layers {@code n_layer-2} and {@code n_layer-1}). */
        public static SharedKv targetLayer(int targetLayer) {
            return new SharedKv(checkLayer(targetLayer), true);
        }

        private static int checkLayer(int il) {
            if (il < 0) {
                throw new IllegalArgumentException("shared KV source layer " + il + " is negative");
            }
            return il;
        }

        public int layer() { return layer; }
        /** True when the layer belongs to the target model bound through {@code ForwardBackend.memory()}. */
        public boolean target() { return target; }

        @Override
        public String toString() {
            return (target ? "TargetLayer(" : "Layer(") + layer + ")";
        }
    }

    /**
     * Lightning-indexer key cache (DSA, QSA k-pool): one row of {@code width} values per KV row in the context's K
     * element type (llama-kv-cache-dsa.cpp:51, llama-memory-hybrid-idx.cpp:71), addressed like the layer's KV group.
     */
    public static final class IndexerPlan {
        private final int rows;
        private final int width;
        private final KvDType dtype;
        private final RotationPlan rotation;

        private IndexerPlan(int rows, int width, KvDType dtype, RotationPlan rotation) {
            this.rows = rows;
            this.width = width;
            this.dtype = dtype;
            this.rotation = rotation;
        }

        /** @throws IllegalArgumentException for non-positive sizes or a width the type cannot store */
        public static IndexerPlan of(int rows, int width, KvDType dtype, RotationPlan rotation) {
            Objects.requireNonNull(dtype, "dtype");
            if (rows <= 0 || width <= 0) {
                throw new IllegalArgumentException("indexer cache needs positive rows and width, got " + rows + "/" + width);
            }
            dtype.rowBytes(width);
            return new IndexerPlan(rows, width, dtype, Objects.requireNonNull(rotation, "rotation"));
        }

        public int rows() { return rows; }
        /** Row width: {@code indexer_head_size}, times {@code kpool_row} for a QSA k-pool indexer. */
        public int width() { return width; }
        /** Element type of the rows. */
        public KvDType dtype() { return dtype; }
        /** Hadamard rotation of indexer q/k. */
        public RotationPlan rotation() { return rotation; }
        public long bytes() { return rows * dtype.rowBytes(width); }

        @Override
        public String toString() {
            return "IndexerPlan{rows=" + rows + ", width=" + width + ", " + dtype + ", " + rotation + '}';
        }
    }

    /** MSA index-key cache (one key of {@code width} per KV row, the main KV type), addressed like the KV group. */
    public static final class MsaPlan {
        private final int rows;
        private final int width;
        private final KvDType dtype;

        private MsaPlan(int rows, int width, KvDType dtype) {
            this.rows = rows;
            this.width = width;
            this.dtype = dtype;
        }

        /** @throws IllegalArgumentException for non-positive sizes or a width the type cannot store */
        public static MsaPlan of(int rows, int width, KvDType dtype) {
            Objects.requireNonNull(dtype, "dtype");
            if (rows <= 0 || width <= 0) {
                throw new IllegalArgumentException("MSA index cache needs positive rows and width, got " + rows + "/" + width);
            }
            dtype.rowBytes(width);
            return new MsaPlan(rows, width, dtype);
        }

        public int rows() { return rows; }
        public int width() { return width; }
        public KvDType dtype() { return dtype; }
        public long bytes() { return rows * dtype.rowBytes(width); }

        @Override
        public String toString() {
            return "MsaPlan{rows=" + rows + ", width=" + width + ", " + dtype + '}';
        }
    }

    /**
     * One DeepSeek-V4 compressed cache with its compressor state (src/llama-kv-cache-dsv4.cpp:1244-1340): a K-only
     * cache of {@code rows = pad256(max(1, ceil(nCtx / ratio)))} rows and two F32 compressor tensors (kv and score),
     * each {@link #planes()} planes of {@code stateRows x stateWidth}, zeroed at reset. Plane 0 is the live state;
     * planes {@code 1..snapshots} are the rollback snapshots ({@code n_rs_seq}, llama-kv-cache-dsv4.cpp:968-970):
     * after a step, plane {@code d} holds the state as it stood before the step's last {@code d} tokens, so dropping
     * those {@code d} positions restores plane {@code d} (llama-kv-cache-dsv4.cpp:657-713, 1463-1510).
     * <ul>
     *   <li>CSA: ratio 4, width {@code head_k}, state {@code 8 x 2 * head_k};</li>
     *   <li>HCA: ratio 128, width {@code head_k}, state {@code 128 x head_k};</li>
     *   <li>INDEXER (lightning indexer of CSA layers): ratio 4, width {@code indexer_head_size}, state
     *       {@code 8 x 2 * indexer_head_size}.</li>
     * </ul>
     */
    public static final class CompressedKvPlan {

        public enum Kind { CSA, HCA, INDEXER }

        private final Kind kind;
        private final int ratio;
        private final int rows;
        private final int width;
        private final KvDType dtype;
        private final int stateRows;
        private final int stateWidth;
        private final int snapshots;

        private CompressedKvPlan(Kind kind, int ratio, int rows, int width, KvDType dtype, int stateRows, int stateWidth,
                                 int snapshots) {
            this.kind = kind;
            this.ratio = ratio;
            this.rows = rows;
            this.width = width;
            this.dtype = dtype;
            this.stateRows = stateRows;
            this.stateWidth = stateWidth;
            this.snapshots = snapshots;
        }

        /**
         * The plan llama.cpp builds for a cache kind, without rollback snapshots.
         *
         * @param headK           {@code n_embd_head_k}
         * @param indexerHeadSize {@code indexer_head_size}
         */
        public static CompressedKvPlan of(Kind kind, int nCtx, int headK, int indexerHeadSize, KvDType dtype) {
            return of(kind, nCtx, headK, indexerHeadSize, dtype, 0);
        }

        /**
         * The plan llama.cpp builds for a cache kind with {@code snapshots} rollback snapshot planes
         * ({@code n_rs_seq}).
         *
         * @param headK           {@code n_embd_head_k}
         * @param indexerHeadSize {@code indexer_head_size}
         * @throws IllegalArgumentException for a non-positive width or negative {@code snapshots}
         */
        public static CompressedKvPlan of(Kind kind, int nCtx, int headK, int indexerHeadSize, KvDType dtype,
                                          int snapshots) {
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(dtype, "dtype");
            int ratio = kind == Kind.HCA ? 128 : 4;
            int width = kind == Kind.INDEXER ? indexerHeadSize : headK;
            if (width <= 0) {
                throw new IllegalArgumentException(kind + " compressed cache needs a positive width, got " + width);
            }
            if (snapshots < 0) {
                throw new IllegalArgumentException(kind + " compressor state needs >= 0 rollback snapshots, got "
                        + snapshots);
            }
            int stateRows = kind == Kind.HCA ? ratio : 2 * ratio;
            int stateWidth = kind == Kind.HCA ? width : 2 * width;
            dtype.rowBytes(width);
            return new CompressedKvPlan(kind, ratio, compressedRows(nCtx, ratio), width, dtype, stateRows, stateWidth,
                    snapshots);
        }

        /** {@code GGML_PAD(max(1, ceil(nCtx / ratio)), 256)}. */
        public static int compressedRows(int nCtx, int ratio) {
            if (nCtx <= 0 || ratio <= 0) {
                throw new IllegalArgumentException("compressed rows need positive nCtx and ratio, got " + nCtx + "/" + ratio);
            }
            long n = Math.max(1L, ((long) nCtx + ratio - 1) / ratio);
            return (int) ((n + KvLayout.RING_PAD - 1) / KvLayout.RING_PAD * KvLayout.RING_PAD);
        }

        public Kind kind() { return kind; }
        public int ratio() { return ratio; }
        public int rows() { return rows; }
        public int width() { return width; }
        public KvDType dtype() { return dtype; }
        public int stateRows() { return stateRows; }
        public int stateWidth() { return stateWidth; }
        /** Rollback snapshot planes ({@code n_rs_seq}): the most trailing positions one rollback can drop. */
        public int snapshots() { return snapshots; }
        /** Planes of each compressor tensor: the live state plus {@link #snapshots()}. */
        public int planes() { return 1 + snapshots; }
        /** Floats of one plane: {@code stateRows * stateWidth}. */
        public int planeFloats() { return stateRows * stateWidth; }
        public long cacheBytes() { return rows * dtype.rowBytes(width); }
        /** Bytes of one compressor tensor (kv or score): {@link #planes()} planes. */
        public long stateTensorBytes() { return (long) planes() * planeFloats() * Float.BYTES; }
        /** Both compressor tensors with every plane. */
        public long stateBytes() { return 2L * stateTensorBytes(); }
        public long bytes() { return cacheBytes() + stateBytes(); }

        @Override
        public String toString() {
            return "CompressedKvPlan{" + kind + ", ratio=" + ratio + ", rows=" + rows + ", width=" + width + ", " + dtype
                    + ", state=" + stateRows + "x" + stateWidth + (snapshots > 0 ? ", snapshots=" + snapshots : "")
                    + '}';
        }
    }

    private final int layer;
    private final Kind kind;
    private final KvLayout kv;
    private final int kvGroup;
    private final RotationPlan rotation;
    private final SharedKv sharedFrom;
    private final List<StateTensor> states;
    private final IndexerPlan indexer;
    private final MsaPlan msa;
    private final List<CompressedKvPlan> compressed;
    private final KvLayout crossKv;

    private LayerMemoryPlan(Builder b) {
        this.layer = b.layer;
        this.kind = b.kind;
        this.kv = b.kv;
        this.kvGroup = b.kvGroup;
        this.rotation = b.rotation;
        this.sharedFrom = b.sharedFrom;
        this.states = Collections.unmodifiableList(new ArrayList<>(b.states));
        this.indexer = b.indexer;
        this.msa = b.msa;
        this.compressed = Collections.unmodifiableList(new ArrayList<>(b.compressed));
        this.crossKv = b.crossKv;
    }

    public static Builder builder(int layer, Kind kind) {
        return new Builder(layer, kind);
    }

    public int layer() { return layer; }
    public Kind kind() { return kind; }
    /** KV layout: own cache, or the owner's layout for SHARED_KV, or the raw window cache for DSV4; null otherwise. */
    public KvLayout kv() { return kv; }
    /** Index of the KV group ({@code kv_pos} buffer) addressing this layer's rows; -1 without KV. */
    public int kvGroup() { return kvGroup; }
    public RotationPlan rotation() { return rotation; }
    /** Cache owner for SHARED_KV; null otherwise. */
    public SharedKv sharedFrom() { return sharedFrom; }
    /** Recurrent state tensors (unmodifiable; empty without state). */
    public List<StateTensor> states() { return states; }
    public IndexerPlan indexer() { return indexer; }
    public MsaPlan msa() { return msa; }
    /** DSV4 compressed caches (unmodifiable; empty otherwise). */
    public List<CompressedKvPlan> compressed() { return compressed; }
    /** T5 decoder cross-attention K/V over the encoder output; null otherwise. */
    public KvLayout crossKv() { return crossKv; }

    public boolean ownsKv() { return kind == Kind.KV || kind == Kind.KV_RECURRENT || kind == Kind.DSV4; }
    public boolean hasState() { return !states.isEmpty(); }

    /** Bytes of the KV cache this layer owns. */
    public long kvBytes() {
        long b = ownsKv() ? kv.totalBytes() : 0L;
        return crossKv != null ? b + crossKv.totalBytes() : b;
    }

    /** Bytes of recurrent state per sequence. */
    public long stateBytes() {
        long b = 0;
        for (StateTensor t : states) {
            b += t.bytes();
        }
        return b;
    }

    /** Bytes of the indexer, MSA and compressed caches. */
    public long sideCacheBytes() {
        long b = indexer != null ? indexer.bytes() : 0L;
        b += msa != null ? msa.bytes() : 0L;
        for (CompressedKvPlan c : compressed) {
            b += c.bytes();
        }
        return b;
    }

    public long totalBytes() {
        return kvBytes() + stateBytes() + sideCacheBytes();
    }

    @Override
    public String toString() {
        return "LayerMemoryPlan{il=" + layer + ", " + kind + (kv != null ? ", " + kv : "")
                + (kvGroup >= 0 ? ", group=" + kvGroup : "")
                + (sharedFrom != null ? ", shared=" + sharedFrom : "") + (states.isEmpty() ? "" : ", states=" + states)
                + (indexer != null ? ", " + indexer : "") + (msa != null ? ", " + msa : "")
                + (compressed.isEmpty() ? "" : ", " + compressed) + '}';
    }

    public static final class Builder {
        private final int layer;
        private final Kind kind;
        private KvLayout kv;
        private int kvGroup = -1;
        private RotationPlan rotation = RotationPlan.NONE;
        private SharedKv sharedFrom;
        private final List<StateTensor> states = new ArrayList<>();
        private IndexerPlan indexer;
        private MsaPlan msa;
        private final List<CompressedKvPlan> compressed = new ArrayList<>();
        private KvLayout crossKv;

        private Builder(int layer, Kind kind) {
            this.layer = layer;
            this.kind = Objects.requireNonNull(kind, "kind");
        }

        public Builder kv(KvLayout layout, int group) { this.kv = layout; this.kvGroup = group; return this; }
        public Builder rotation(RotationPlan v) { this.rotation = Objects.requireNonNull(v, "rotation"); return this; }
        public Builder sharedFrom(SharedKv v) { this.sharedFrom = v; return this; }
        public Builder state(StateTensor v) { this.states.add(Objects.requireNonNull(v, "state")); return this; }
        public Builder states(List<StateTensor> v) {
            for (StateTensor t : v) {
                state(t);
            }
            return this;
        }
        public Builder indexer(IndexerPlan v) { this.indexer = v; return this; }
        public Builder msa(MsaPlan v) { this.msa = v; return this; }
        public Builder compressed(CompressedKvPlan v) {
            this.compressed.add(Objects.requireNonNull(v, "compressed"));
            return this;
        }
        public Builder crossKv(KvLayout v) { this.crossKv = v; return this; }

        /** @throws IllegalArgumentException when the parts present do not match the kind */
        public LayerMemoryPlan build() {
            if (layer < 0) {
                throw new IllegalArgumentException("memory plan layer " + layer + " is negative");
            }
            boolean hasKv = kv != null;
            boolean hasStates = !states.isEmpty();
            switch (kind) {
                case NONE:
                    check(!hasKv && !hasStates && sharedFrom == null && indexer == null && msa == null && compressed.isEmpty(),
                            "no memory");
                    break;
                case KV:
                    check(hasKv && !hasStates && sharedFrom == null && compressed.isEmpty(), "a KV layout only");
                    break;
                case SHARED_KV:
                    check(sharedFrom != null && !hasStates && indexer == null && msa == null && compressed.isEmpty(),
                            "a shared-KV source and no owned caches");
                    check(sharedFrom.target() || sharedFrom.layer() != layer, "a source layer other than itself");
                    break;
                case RECURRENT:
                    check(!hasKv && hasStates && sharedFrom == null && indexer == null && msa == null && compressed.isEmpty(),
                            "recurrent state only");
                    break;
                case KV_RECURRENT:
                    check(hasKv && hasStates && sharedFrom == null && compressed.isEmpty(), "a KV layout and recurrent state");
                    break;
                case DSV4:
                    check(hasKv && kv.window().windowed() && sharedFrom == null && !hasStates, "a raw sliding-window KV layout");
                    break;
                default:
                    throw new IllegalStateException("unhandled memory kind " + kind);
            }
            check(crossKv == null || kind == Kind.KV, "cross-attention K/V only on a decoder KV layer");
            check(crossKv == null || !crossKv.ring(), "a full (non-ring) cross-attention cache");
            check(hasKv == (kvGroup >= 0), "a KV group exactly when a KV layout is present");
            check(indexer == null || (hasKv && indexer.rows() == kv.rows()), "indexer rows equal to the KV rows");
            check(msa == null || (hasKv && msa.rows() == kv.rows()), "MSA rows equal to the KV rows");
            return new LayerMemoryPlan(this);
        }

        private void check(boolean ok, String what) {
            if (!ok) {
                throw new IllegalArgumentException("memory plan of layer " + layer + " (" + kind + ") needs " + what);
            }
        }
    }
}
