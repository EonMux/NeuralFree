package com.eonmux.neuralfree.gguf.memory;

import java.util.Objects;

/**
 * Row layout of one KV cache: {@link #rows()} rows of K ({@link #widthK()} values per token across all KV heads)
 * and V ({@link #widthV()} values). A position is stored in row {@link #rowOf(int)}: {@code pos % rows} for a ring
 * (sliding-window layers) and {@code pos} otherwise. Within a row, head {@code h} occupies values
 * {@code [h * headDim, (h + 1) * headDim)}.
 *
 * <p>With {@link #vAliasesK()} (MLA latent cache) no V buffer exists: V is the first {@link #widthV()} values of
 * each K row ({@code d_v = kv_lora}). A per-token-head dtype adds one F32 {@link #sideBytes() side plane} per cache,
 * one scale per (row, KV head); {@link #nHeadKv()} sizes those planes. Immutable.
 */
public final class KvLayout {

    /** Ring rows are padded to this multiple ({@code GGML_PAD(..., 256)}, llama-kv-cache-iswa.cpp:73). */
    public static final int RING_PAD = 256;

    private final int rows;
    private final boolean ring;
    private final int widthK;
    private final int widthV;
    private final KvDType dtypeK;
    private final KvDType dtypeV;
    private final boolean vAliasesK;
    private final KvWindow window;
    private final int nHeadKv;
    private final int rowBase;
    private final boolean sharded;

    private KvLayout(int rows, boolean ring, int widthK, int widthV, KvDType dtypeK, KvDType dtypeV, boolean vAliasesK,
                     KvWindow window, int nHeadKv) {
        this(rows, ring, widthK, widthV, dtypeK, dtypeV, vAliasesK, window, nHeadKv, 0, false);
    }

    private KvLayout(int rows, boolean ring, int widthK, int widthV, KvDType dtypeK, KvDType dtypeV, boolean vAliasesK,
                     KvWindow window, int nHeadKv, int rowBase, boolean sharded) {
        this.rows = rows;
        this.ring = ring;
        this.widthK = widthK;
        this.widthV = widthV;
        this.dtypeK = dtypeK;
        this.dtypeV = dtypeV;
        this.vAliasesK = vAliasesK;
        this.window = window;
        this.nHeadKv = nHeadKv;
        this.rowBase = rowBase;
        this.sharded = sharded;
    }

    /**
     * Layout with separate K and V rows.
     *
     * @throws IllegalArgumentException for non-positive sizes, widths a dtype cannot store, or a ring that cannot
     *                                  hold its window
     */
    public static KvLayout of(int rows, boolean ring, int widthK, int widthV, KvDType dtypeK, KvDType dtypeV,
                              KvWindow window) {
        return create(rows, ring, widthK, widthV, dtypeK, dtypeV, false, window);
    }

    /**
     * MLA latent layout: K rows of {@code widthK = kv_lora + n_rot}; V aliases the first {@code widthV = kv_lora}
     * values of each K row.
     */
    public static KvLayout latent(int rows, boolean ring, int widthK, int widthV, KvDType dtype, KvWindow window) {
        if (widthV > widthK) {
            throw new IllegalArgumentException("latent V width " + widthV + " exceeds the K row width " + widthK);
        }
        return create(rows, ring, widthK, widthV, dtype, dtype, true, window);
    }

    private static KvLayout create(int rows, boolean ring, int widthK, int widthV, KvDType dtypeK, KvDType dtypeV,
                                   boolean vAliasesK, KvWindow window) {
        Objects.requireNonNull(dtypeK, "dtypeK");
        Objects.requireNonNull(dtypeV, "dtypeV");
        Objects.requireNonNull(window, "window");
        if (rows <= 0 || widthK <= 0 || widthV <= 0) {
            throw new IllegalArgumentException("KV layout needs positive rows and widths, got rows=" + rows + " K=" + widthK
                    + " V=" + widthV);
        }
        dtypeK.rowBytes(widthK);
        if (!vAliasesK) {
            dtypeV.rowBytes(widthV);
        }
        if (ring) {
            if (!window.windowed() || !window.causal()) {
                throw new IllegalArgumentException("a ring KV cache needs a causal sliding window, got " + window);
            }
            if (rows < window.nSwa()) {
                throw new IllegalArgumentException("ring of " + rows + " rows cannot hold a window of " + window.nSwa());
            }
        }
        return new KvLayout(rows, ring, widthK, widthV, dtypeK, dtypeV, vAliasesK, window, 1);
    }

    /**
     * This layout with {@code nHeadKv} KV heads per row: the count the per-token-head scale planes size themselves
     * by ({@link SidePlane#widthFor(int)}). The row widths do not change; a per-token-head cache tiles them with its
     * heads, which the store validates.
     *
     * @throws IllegalArgumentException for a non-positive head count
     */
    public KvLayout withHeads(int nHeadKv) {
        if (nHeadKv <= 0) {
            throw new IllegalArgumentException("KV layout needs a positive KV head count, got " + nHeadKv);
        }
        return new KvLayout(rows, ring, widthK, widthV, dtypeK, dtypeV, vAliasesK, window, nHeadKv, rowBase, sharded);
    }

    /**
     * This layout as one rank's context shard (vLLM's decode context parallelism): row {@code r} holds the position
     * {@code rowBase + r}, and the shard keeps the {@code rows} rows of the block
     * {@code [rowBase, rowBase + rows)}. A shard never wraps.
     *
     * @throws IllegalArgumentException for a negative base, or a ring
     */
    public KvLayout withShard(int rowBase) {
        if (rowBase < 0) {
            throw new IllegalArgumentException("a context shard needs a non-negative row base, got " + rowBase);
        }
        if (ring) {
            throw new IllegalArgumentException("a context shard cannot be a ring");
        }
        return new KvLayout(rows, ring, widthK, widthV, dtypeK, dtypeV, vAliasesK, window, nHeadKv, rowBase, true);
    }

    /** Global position of row 0 of a context shard; 0 for a whole cache. */
    public int rowBase() { return rowBase; }

    /** True when this layout is one rank's context shard: its rows cover {@code [rowBase, rowBase + rows)}. */
    public boolean sharded() { return sharded; }

    /**
     * Ring size for a sliding-window layer: {@code nCtx} when {@code swaFull}, otherwise
     * {@code min(nCtx, roundUp(nSwa + ubatch, 256))}.
     */
    public static int ringRows(int nCtx, int nSwa, int ubatch, boolean swaFull) {
        if (nCtx <= 0 || nSwa <= 0 || ubatch <= 0) {
            throw new IllegalArgumentException("ring sizing needs positive nCtx, nSwa and ubatch, got " + nCtx + "/" + nSwa
                    + "/" + ubatch);
        }
        if (swaFull) {
            return nCtx;
        }
        long need = (long) nSwa + ubatch;
        long padded = (need + RING_PAD - 1) / RING_PAD * RING_PAD;
        return (int) Math.min(nCtx, padded);
    }

    public int rows() { return rows; }
    /** True when positions wrap around ({@code rows < nCtx}). */
    public boolean ring() { return ring; }
    public int widthK() { return widthK; }
    public int widthV() { return widthV; }
    public KvDType dtypeK() { return dtypeK; }
    public KvDType dtypeV() { return dtypeV; }
    /** V is read from the K rows (MLA latent cache); there is no V buffer. */
    public boolean vAliasesK() { return vAliasesK; }
    public KvWindow window() { return window; }

    /** KV heads per row: 1 for a latent or single-head cache. Sizes the per-token-head scale planes. */
    public int nHeadKv() { return nHeadKv; }

    /** The side-plane rows of K of the dtype ({@link #SCALE_PLANE} included); 0 for a dtype without planes. */
    public long sideBytesK() {
        return sideBytes(dtypeK, widthK);
    }

    /** The side-plane rows of V; 0 when V aliases K or the V dtype keeps no side plane. */
    public long sideBytesV() {
        return vAliasesK ? 0L : sideBytes(dtypeV, widthV);
    }

    /** Bytes of the side planes of one dtype over every row. */
    private long sideBytes(KvDType dtype, int width) {
        long b = 0;
        for (KvDType.SidePlane plane : dtype.sidePlanes()) {
            b += plane.rowBytesFor(width, nHeadKv);
        }
        return b * rows;
    }

    /** Bytes of the side planes of K and (separate) V together. */
    public long sideBytes() {
        return sideBytesK() + sideBytesV();
    }

    /** Bytes of one K row. */
    public long rowBytesK() {
        return dtypeK.rowBytes(widthK);
    }

    /** Bytes of one V row; the K row size when V aliases K. */
    public long rowBytesV() {
        return vAliasesK ? rowBytesK() : dtypeV.rowBytes(widthV);
    }

    /** Bytes of the K buffer. */
    public long kBytes() {
        return rows * rowBytesK();
    }

    /** Bytes of the V buffer; 0 when V aliases K. */
    public long vBytes() {
        return vAliasesK ? 0L : rows * rowBytesV();
    }

    /** Bytes of K and V together, with the side planes their dtypes keep beside the rows. */
    public long totalBytes() {
        return kBytes() + vBytes() + sideBytes();
    }

    /**
     * Row holding absolute position {@code pos}; on a context shard only the block's positions have a row.
     *
     * @throws IllegalArgumentException for a negative position, a position beyond a non-ring cache, or a position
     *                                  outside a context shard
     */
    public int rowOf(int pos) {
        if (pos < 0) {
            throw new IllegalArgumentException("KV position " + pos + " is negative");
        }
        if (ring) {
            return pos % rows;
        }
        if (sharded) {
            if (pos < rowBase || pos >= rowBase + rows) {
                throw new IllegalArgumentException("KV position " + pos + " is outside the context shard of rows ["
                        + rowBase + ", " + (rowBase + rows) + ")");
            }
            return pos - rowBase;
        }
        if (pos >= rows) {
            throw new IllegalArgumentException("KV position " + pos + " is beyond the cache of " + rows + " rows");
        }
        return pos;
    }

    /** True when the cache, or the context shard, stores position {@code pos}. */
    public boolean holds(int pos) {
        if (pos < 0) {
            return false;
        }
        return ring || (sharded ? pos >= rowBase && pos < rowBase + rows : pos < rows);
    }

    @Override
    public String toString() {
        return "KvLayout{rows=" + rows + (ring ? " ring" : "") + (sharded ? " shard@" + rowBase : "")
                + ", K=" + widthK + " " + dtypeK + ", V=" + widthV + " "
                + (vAliasesK ? "aliases K" : dtypeV.toString()) + (nHeadKv != 1 ? ", heads=" + nHeadKv : "") + ", "
                + window + '}';
    }
}
