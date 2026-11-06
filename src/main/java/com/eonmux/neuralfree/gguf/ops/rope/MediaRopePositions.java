package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * RoPE positions of one media chunk's embedding rows, the Java counterpart of llama.cpp
 * {@code mtmd_image_tokens_get_decoder_pos} (tools/mtmd/mtmd.cpp:3078) and
 * {@code decode_embd_batch.set_position_mrope_2d / _1d / _normal} (tools/mtmd/mtmd-helper-common.h:101-126).
 * Positions are laid out section-major, {@code [s][t]} (the {@code ForwardBatch.ropePositions} and
 * {@code GpuStepInputs.ropePos} layout): section 0 carries the temporal coordinate t, sections 1 and 2 the image
 * grid coordinates y and x, section 3 the reserved z.
 *
 * <p>The decoder positions of an image row {@code i} of an {@code nx * ny} grid at base position {@code pos0} are
 * {@code t = pos0}, {@code y = pos0 + i / nx}, {@code x = pos0 + i % nx}, {@code z = 0}; the whole chunk occupies
 * {@code max(nx, ny)} temporal positions ({@link #nPositions()}), so the rows after it start at
 * {@code pos0 + max(nx, ny)}, not after {@code nTokens}. Audio rows carry {@code pos0 + i} in every section, and
 * normal (non M-RoPE) rows likewise, occupying one position per row. SenseNova-U1 image rows
 * ({@code MTMD_POS_TYPE_SENSENOVA_U1}) take {@code t = pos0}, {@code y = i / nx}, {@code x = i % nx}, {@code z = 0}:
 * the grid coordinates count from 0 and the whole image occupies one temporal position. A MiniCPM-V 4.7 canvas
 * chunk ({@link #canvas}) lays the overview and every slice on one 2D canvas.</p>
 *
 * <p>Immutable; arrays are copied at construction and accessors return copies.</p>
 */
public final class MediaRopePositions {

    /** How the rows of a media chunk take their positions. */
    public enum Kind {
        /** M-RoPE image grid: t shared, y and x from the grid cell, z 0. */
        IMAGE_2D,
        /** SenseNova-U1 image grid: t shared, y and x from the grid cell counted from 0, z 0; one position. */
        IMAGE_2D_ZERO_BASED,
        /** M-RoPE audio or video frames: every section advances by one per row. */
        AUDIO_1D,
        /** One position per row in every section (non M-RoPE models). */
        NORMAL,
        /** M-RoPE MiniCPM-V 4.7 canvas: the overview and the slices on one 2D canvas, separators between them. */
        CANVAS
    }

    /** One tile of a {@link #canvas} chunk: its token grid and the separator rows after it. Immutable. */
    public static final class CanvasTile {
        private final int tokensX;
        private final int tokensY;
        private final int separators;

        /**
         * A {@code tokensX x tokensY} token grid followed by {@code separators} rows.
         *
         * @throws IllegalArgumentException when a side is not positive or the separator count is negative
         */
        public CanvasTile(int tokensX, int tokensY, int separators) {
            if (tokensX <= 0 || tokensY <= 0 || separators < 0) {
                throw new IllegalArgumentException("a canvas tile needs a positive token grid and no negative"
                        + " separator count, got " + tokensX + "x" + tokensY + " + " + separators);
            }
            this.tokensX = tokensX;
            this.tokensY = tokensY;
            this.separators = separators;
        }

        /** Tokens per row. */
        public int tokensX() {
            return tokensX;
        }

        /** Token rows. */
        public int tokensY() {
            return tokensY;
        }

        /** Separator rows after the tile's tokens. */
        public int separators() {
            return separators;
        }

        int rows() {
            return Math.addExact(Math.multiplyExact(tokensX, tokensY), separators);
        }
    }

    private final Kind kind;
    private final int pos0;
    private final int nTokens;
    private final int nx;
    private final int ny;
    private final int nPositions;
    private final int[][] sections; // [4][nTokens]

    private MediaRopePositions(Kind kind, int pos0, int nTokens, int nx, int ny, int nPositions, int[][] sections) {
        this.kind = kind;
        this.pos0 = pos0;
        this.nTokens = nTokens;
        this.nx = nx;
        this.ny = ny;
        this.nPositions = nPositions;
        this.sections = sections;
    }

    /**
     * Positions of an {@code nx * ny} image grid of {@code nx * ny} embedding rows whose first row sits at
     * {@code pos0}.
     *
     * @throws IllegalArgumentException when a count is not positive or {@code pos0} is negative
     */
    public static MediaRopePositions image2d(int pos0, int nx, int ny) {
        if (nx <= 0 || ny <= 0) {
            throw new IllegalArgumentException("image grid needs positive nx and ny, got " + nx + " x " + ny);
        }
        return build(Kind.IMAGE_2D, pos0, Math.multiplyExact(nx, ny), nx, ny);
    }

    /**
     * Positions of the kept cells of an {@code nx * ny} image grid: {@code keep[t]} says whether raster cell
     * {@code t} has its row (the retention mask of a pruned video chunk, vLLM's EVS and VidCom2 keepers); the
     * kept rows take their own cells' positions in order, and the chunk still occupies {@code max(nx, ny)}
     * positions, as the unpruned grid does.
     *
     * @throws IllegalArgumentException when {@code keep} does not hold one entry per cell or keeps no cell
     */
    public static MediaRopePositions image2dMasked(int pos0, int nx, int ny, boolean[] keep) {
        if (nx <= 0 || ny <= 0) {
            throw new IllegalArgumentException("image grid needs positive nx and ny, got " + nx + " x " + ny);
        }
        Objects.requireNonNull(keep, "keep");
        if (keep.length != Math.multiplyExact(nx, ny)) {
            throw new IllegalArgumentException("a retention mask of " + keep.length + " entries does not cover the "
                    + nx + "x" + ny + " grid of " + (nx * ny) + " cells");
        }
        int nTokens = 0;
        for (boolean k : keep) {
            if (k) {
                nTokens++;
            }
        }
        if (nTokens == 0) {
            throw new IllegalArgumentException("a retention mask keeps no cell");
        }
        if (nTokens == keep.length) {
            return build(Kind.IMAGE_2D, pos0, nTokens, nx, ny);
        }
        int[][] s = new int[MropeTextPositions.SECTIONS][nTokens];
        int i = 0;
        for (int t = 0; t < keep.length; t++) {
            if (!keep[t]) {
                continue;
            }
            s[0][i] = pos0;
            s[1][i] = pos0 + t / nx;
            s[2][i] = pos0 + t % nx;
            s[3][i] = 0;
            i++;
        }
        int nPos = Math.max(nx, ny);
        return new MediaRopePositions(Kind.IMAGE_2D, pos0, nTokens, nx, ny, nPos, s);
    }

    /**
     * Positions of an {@code nx * ny} SenseNova-U1 image grid whose rows sit at {@code pos0}: t is {@code pos0},
     * y and x the grid cell counted from 0, z 0; the chunk occupies one position.
     *
     * @throws IllegalArgumentException when a count is not positive or {@code pos0} is negative
     */
    public static MediaRopePositions image2dZeroBased(int pos0, int nx, int ny) {
        if (nx <= 0 || ny <= 0) {
            throw new IllegalArgumentException("image grid needs positive nx and ny, got " + nx + " x " + ny);
        }
        return build(Kind.IMAGE_2D_ZERO_BASED, pos0, Math.multiplyExact(nx, ny), nx, ny);
    }

    /**
     * Positions of {@code nTokens} audio (or other sequential-media) rows starting at {@code pos0}: every section
     * advances by one per row, as {@code set_position_mrope_1d} fills them.
     *
     * @throws IllegalArgumentException when {@code nTokens} is not positive or {@code pos0} is negative
     */
    public static MediaRopePositions audio1d(int pos0, int nTokens) {
        if (nTokens <= 0) {
            throw new IllegalArgumentException("media rows need a positive count, got " + nTokens);
        }
        return build(Kind.AUDIO_1D, pos0, nTokens, 1, 1);
    }

    /**
     * Positions of {@code nTokens} rows at {@code pos0 ..} on a model without M-RoPE: every section carries the
     * row position, as {@code set_position_normal} fills them.
     *
     * @throws IllegalArgumentException when {@code nTokens} is not positive or {@code pos0} is negative
     */
    public static MediaRopePositions normal(int pos0, int nTokens) {
        if (nTokens <= 0) {
            throw new IllegalArgumentException("media rows need a positive count, got " + nTokens);
        }
        return build(Kind.NORMAL, pos0, nTokens, 1, 1);
    }

    private static MediaRopePositions build(Kind kind, int pos0, int nTokens, int nx, int ny) {
        if (pos0 < 0) {
            throw new IllegalArgumentException("media base position " + pos0 + " is negative");
        }
        int[][] s = new int[MropeTextPositions.SECTIONS][nTokens];
        for (int i = 0; i < nTokens; i++) {
            int t;
            int y;
            int x;
            int z;
            switch (kind) {
                case IMAGE_2D:
                    t = pos0;
                    x = pos0 + i % nx;
                    y = pos0 + i / nx;
                    z = 0;
                    break;
                case IMAGE_2D_ZERO_BASED:
                    t = pos0;
                    x = i % nx;
                    y = i / nx;
                    z = 0;
                    break;
                case AUDIO_1D:
                case NORMAL:
                default:
                    t = pos0 + i;
                    x = pos0 + i;
                    y = pos0 + i;
                    z = pos0 + i;
                    break;
            }
            s[0][i] = t;
            s[1][i] = y;
            s[2][i] = x;
            s[3][i] = z;
        }
        int nPos = kind == Kind.IMAGE_2D ? Math.max(nx, ny) : kind == Kind.IMAGE_2D_ZERO_BASED ? 1 : nTokens;
        return new MediaRopePositions(kind, pos0, nTokens, nx, ny, nPos, s);
    }

    /**
     * Positions of a MiniCPM-V 4.7 canvas chunk ({@code MTMD_POS_TYPE_CANVAS}, {@code mtmd_canvas_decoder_pos}):
     * the overview and the {@code gridX x gridY} slices share the canvas of the slice grid, {@code gridX * nx} by
     * {@code gridY * ny} tokens with {@code nx x ny} the last tile's grid (the overview's own grid when there are
     * no slices). The {@code <image>} token at {@code pos0 - 1} is the origin {@code base}. Tile {@code k} takes
     * the temporal position {@code t_k}, starting at {@code pos0}; its token {@code (row, col)} sits at
     * {@code y = base + h}, {@code x = base + w}, {@code z = base}, where the overview spreads over the whole canvas
     * ({@code h = round(row * (H - 1) / (gh - 1))}, ties to even, likewise {@code w}) and slice {@code s} fills its
     * cell ({@code h = (s / gridX) * ny + row}, {@code w = (s % gridX) * nx + col}). Separator {@code j} after tile
     * {@code k} takes {@code t_k + 1 + j} in every section, and the next tile starts after them. The chunk occupies
     * one position per tile plus one per separator.
     *
     * @param pos0  the position of the first row; the {@code <image>} token before the chunk sits at {@code pos0 - 1}
     * @param gridX slices per row, 0 for an overview alone
     * @param gridY slice rows, 0 for an overview alone
     * @param tiles the overview, then the slices row by row
     * @throws IllegalArgumentException when {@code pos0} is not positive, the grid is negative or half empty, or the
     *                                  tiles are not one overview plus {@code gridX * gridY} slices
     */
    public static MediaRopePositions canvas(int pos0, int gridX, int gridY, List<CanvasTile> tiles) {
        Objects.requireNonNull(tiles, "tiles");
        if (pos0 < 1) {
            throw new IllegalArgumentException("a canvas chunk follows its <image> token, so its base position must"
                    + " be positive, got " + pos0);
        }
        if (gridX < 0 || gridY < 0 || (gridX == 0) != (gridY == 0)
                || tiles.size() != 1 + Math.multiplyExact(gridX, gridY)) {
            throw new IllegalArgumentException("a canvas chunk holds the overview and a " + gridX + "x" + gridY
                    + " slice grid, got " + tiles.size() + " tiles");
        }
        CanvasTile ref = tiles.get(tiles.size() - 1);
        int nx = ref.tokensX();
        int ny = ref.tokensY();
        int canvasW = gridX > 0 ? Math.multiplyExact(gridX, nx) : nx;
        int canvasH = gridY > 0 ? Math.multiplyExact(gridY, ny) : ny;
        int nTokens = 0;
        int nPos = 0;
        for (CanvasTile tile : tiles) {
            nTokens = Math.addExact(nTokens, tile.rows());
            nPos = Math.addExact(nPos, 1 + tile.separators());
        }
        int[][] s = new int[MropeTextPositions.SECTIONS][nTokens];
        int base = pos0 - 1;
        int t = pos0;
        int i = 0;
        for (int k = 0; k < tiles.size(); k++) {
            CanvasTile tile = tiles.get(k);
            for (int row = 0; row < tile.tokensY(); row++) {
                for (int col = 0; col < tile.tokensX(); col++, i++) {
                    int h = k == 0 ? canvasScale(row, tile.tokensY(), canvasH) : ((k - 1) / gridX) * ny + row;
                    int w = k == 0 ? canvasScale(col, tile.tokensX(), canvasW) : ((k - 1) % gridX) * nx + col;
                    s[0][i] = t;
                    s[1][i] = base + h;
                    s[2][i] = base + w;
                    s[3][i] = base;
                }
            }
            for (int j = 0; j < tile.separators(); j++, i++) {
                for (int[] section : s) {
                    section[i] = t + 1 + j;
                }
            }
            t += 1 + tile.separators();
        }
        return new MediaRopePositions(Kind.CANVAS, pos0, nTokens, nx, ny, nPos, s);
    }

    /**
     * {@code mtmd_canvas_scale}: grid coordinate {@code coord} of {@code grid} spread over {@code canvas} cells,
     * {@code round(linspace(0, canvas - 1, grid))} with ties to even.
     */
    private static int canvasScale(int coord, int grid, int canvas) {
        if (grid <= 1 || canvas <= 1) {
            return 0;
        }
        double v = (double) coord * (canvas - 1) / (grid - 1);
        return (int) Math.min(Math.rint(v), canvas - 1);
    }

    /** The position kind of the rows. */
    public Kind kind() {
        return kind;
    }

    /** Base position ({@code pos_0}) of the first row. */
    public int pos0() {
        return pos0;
    }

    /** Rows the chunk holds ({@code mtmd_input_chunk_get_n_tokens}). */
    public int nTokens() {
        return nTokens;
    }

    /** Grid width ({@code nx}; the last tile's for a canvas); 1 for non-grid kinds. */
    public int nx() {
        return nx;
    }

    /** Grid height ({@code ny}; the last tile's for a canvas); 1 for non-grid kinds. */
    public int ny() {
        return ny;
    }

    /**
     * Temporal positions the chunk occupies ({@code mtmd_input_chunk_get_n_pos}): {@code max(nx, ny)} for an
     * image grid, one for a SenseNova-U1 grid, one per tile and separator for a canvas, one per row otherwise.
     * The rows after the chunk start at {@code pos0 + nPositions()}.
     */
    public int nPositions() {
        return nPositions;
    }

    /**
     * Section-major positions of every row: {@code out[s]} holds section {@code s} ({@code 0 = t, 1 = y, 2 = x,
     * 3 = z}) of all rows.
     */
    public int[][] sections() {
        int[][] out = new int[sections.length][];
        for (int s = 0; s < sections.length; s++) {
            out[s] = sections[s].clone();
        }
        return out;
    }

    /** Section {@code s} of every row (a copy). */
    public int[] section(int s) {
        Objects.checkIndex(s, sections.length);
        return sections[s].clone();
    }

    /** Position of row {@code i} in section {@code s}. */
    public int position(int s, int i) {
        Objects.checkIndex(s, sections.length);
        Objects.checkIndex(i, nTokens);
        return sections[s][i];
    }

    /** Two position sets describe the same kind, base, grid and per-row positions. */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof MediaRopePositions)) {
            return false;
        }
        MediaRopePositions other = (MediaRopePositions) o;
        return kind == other.kind && pos0 == other.pos0 && nTokens == other.nTokens
                && nPositions == other.nPositions && Arrays.deepEquals(sections, other.sections);
    }

    @Override
    public int hashCode() {
        int h = Objects.hash(kind, pos0, nTokens, nPositions);
        h = 31 * h + Arrays.deepHashCode(sections);
        return h;
    }

    @Override
    public String toString() {
        return "MediaRopePositions{" + kind + ", pos0=" + pos0 + ", tokens=" + nTokens + ", grid=" + nx + "x" + ny
                + ", nPos=" + nPositions() + '}';
    }
}
