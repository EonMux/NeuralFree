package com.eonmux.neuralfree.gguf.memory;

import java.util.Objects;

/**
 * Row layout of one KV cache: {@link #rows()} rows of K ({@link #widthK()} values per token across all KV heads)
 * and V ({@link #widthV()} values). A position is stored in row {@link #rowOf(int)}: {@code pos % rows} for a ring
 * (sliding-window layers) and {@code pos} otherwise. Within a row, head {@code h} occupies values
 * {@code [h * headDim, (h + 1) * headDim)}.
 *
 * <p>With {@link #vAliasesK()} (MLA latent cache) no V buffer exists: V is the first {@link #widthV()} values of
 * each K row ({@code d_v = kv_lora}). Immutable.
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

    private KvLayout(int rows, boolean ring, int widthK, int widthV, KvDType dtypeK, KvDType dtypeV, boolean vAliasesK,
                     KvWindow window) {
        this.rows = rows;
        this.ring = ring;
        this.widthK = widthK;
        this.widthV = widthV;
        this.dtypeK = dtypeK;
        this.dtypeV = dtypeV;
        this.vAliasesK = vAliasesK;
        this.window = window;
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
        return new KvLayout(rows, ring, widthK, widthV, dtypeK, dtypeV, vAliasesK, window);
    }

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

    /** Bytes of K and V together. */
    public long totalBytes() {
        return kBytes() + vBytes();
    }

    /**
     * Row holding absolute position {@code pos}.
     *
     * @throws IllegalArgumentException for a negative position, or a position beyond a non-ring cache
     */
    public int rowOf(int pos) {
        if (pos < 0) {
            throw new IllegalArgumentException("KV position " + pos + " is negative");
        }
        if (ring) {
            return pos % rows;
        }
        if (pos >= rows) {
            throw new IllegalArgumentException("KV position " + pos + " is beyond the cache of " + rows + " rows");
        }
        return pos;
    }

    @Override
    public String toString() {
        return "KvLayout{rows=" + rows + (ring ? " ring" : "") + ", K=" + widthK + " " + dtypeK + ", V=" + widthV + " "
                + (vAliasesK ? "aliases K" : dtypeV.toString()) + ", " + window + '}';
    }
}
