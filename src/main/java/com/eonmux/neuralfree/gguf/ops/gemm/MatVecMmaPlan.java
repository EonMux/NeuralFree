package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * Immutable launch plan of the tensor-core multi-column matvec {@code {T}_MatVecMma{8,16}_Q8}
 * (kernels/llm/matvec_mma.cl): {@code n} token rows of 1..{@value #MAX_TOKENS} through a quantized weight of
 * {@code rows x cols}, against the {@code n} consecutive Q8_1 activation rows the counted matvec lane reads,
 * writing {@code out[t * rows + r]}.
 *
 * <p>A work-group owns {@link #tileRows} weight rows (one or two m16 tiles, per type and token tile as the
 * kernel file instantiates them) and splits the reduction over its {@code warps} warps (1..{@value #MAX_WARPS},
 * chosen by the caller from measurements); the weight streams from DRAM straight into the mma fragments. The
 * token tile is 8 up to 8 tokens and 16 above.</p>
 *
 * <p>Served shapes: {@link GGMLType#Q8_0} with {@code cols % 64 == 0} (a whole number of word-aligned block
 * pairs per row) and {@link GGMLType#Q4_K} / {@link GGMLType#Q6_K} with {@code cols % 512 == 0} (an even block
 * count keeps every row word aligned); any row count.</p>
 */
public final class MatVecMmaPlan {

    /** Largest warp count of a work-group (the kernel's shared reduction buffer). */
    public static final int MAX_WARPS = 8;
    /** Largest token count (two n8 tiles). */
    public static final int MAX_TOKENS = 16;

    private final String kernel;
    private final int rows;
    private final int cols;
    private final int tokens;
    private final int warps;
    private final int tileRows;

    private MatVecMmaPlan(String kernel, int rows, int cols, int tokens, int warps, int tileRows) {
        this.kernel = kernel;
        this.rows = rows;
        this.cols = cols;
        this.tokens = tokens;
        this.warps = warps;
        this.tileRows = tileRows;
    }

    /** {@code {T}_MatVecMma{tile}_Q8}. */
    public static String kernelName(GGMLType type, int tile) {
        checkTile(tile);
        return type.name() + "_MatVecMma" + tile + "_Q8";
    }

    /** Whether the kernel family serves weight type {@code type} with {@code cols} columns. */
    public static boolean serves(GGMLType type, int cols) {
        if (type == null || cols <= 0) {
            return false;
        }
        switch (type) {
            case Q8_0:
                return cols % 64 == 0;
            case Q4_K:
            case Q6_K:
                return cols % 512 == 0;
            default:
                return false;
        }
    }

    /** The token tile of an {@code n}-token launch: 8 up to 8 tokens, 16 above. */
    public static int tileTokens(int n) {
        return n <= 8 ? 8 : 16;
    }

    /**
     * The weight rows one work-group of {@code {T}_MatVecMma{tile}_Q8} owns: 16 for the Q8_0 16-token form
     * (one m16 tile, the activation words loaded a step ahead), 32 (two m16 tiles) for every other form.
     */
    public static int tileRows(GGMLType type, int tile) {
        checkTile(tile);
        return type == GGMLType.Q8_0 && tile == 16 ? 16 : 32;
    }

    /**
     * The plan of {@code n} tokens over a {@code rows x cols} weight of {@code type} at {@code warps} warps per
     * work-group.
     *
     * @throws IllegalArgumentException when the type and shape are not served, {@code n} is outside
     *                                  [1, {@value #MAX_TOKENS}] or {@code warps} outside [1, {@value #MAX_WARPS}]
     */
    public static MatVecMmaPlan of(GGMLType type, int rows, int cols, int n, int warps) {
        if (!serves(type, cols)) {
            throw new IllegalArgumentException("the tensor-core matvec does not serve " + type + " with " + cols
                    + " columns");
        }
        if (rows <= 0) {
            throw new IllegalArgumentException("rows must be positive, got " + rows);
        }
        if (n < 1 || n > MAX_TOKENS) {
            throw new IllegalArgumentException("token count must be in [1, " + MAX_TOKENS + "], got " + n);
        }
        if (warps < 1 || warps > MAX_WARPS) {
            throw new IllegalArgumentException("warps must be in [1, " + MAX_WARPS + "], got " + warps);
        }
        int tile = tileTokens(n);
        return new MatVecMmaPlan(kernelName(type, tile), rows, cols, n, warps, tileRows(type, tile));
    }

    private static void checkTile(int tile) {
        if (tile != 8 && tile != 16) {
            throw new IllegalArgumentException("token tile must be 8 or 16, got " + tile);
        }
    }

    public String kernel() {
        return kernel;
    }

    public int warps() {
        return warps;
    }

    public int tileRows() {
        return tileRows;
    }

    public long[] globalWorkSize() {
        return new long[]{(((long) rows + tileRows - 1) / tileRows) * 32L * warps};
    }

    public long[] localWorkSize() {
        return new long[]{32L * warps};
    }

    /** The scalar arguments in kernel order: rows, cols, n_tokens. */
    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public int tokens() {
        return tokens;
    }
}
