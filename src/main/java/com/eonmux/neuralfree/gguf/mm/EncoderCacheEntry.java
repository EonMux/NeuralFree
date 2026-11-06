package com.eonmux.neuralfree.gguf.mm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The encoder output of one media item, in the layout the media context appends it (vLLM's encoder cache entry for
 * one {@code mm_hash}): the item's marker text pieces and its embedding-row pieces, each carrying the token grid, the
 * canvas tiles or the lead pads the prompt lays it out with.
 *
 * <p>One chunk is one {@link MediaContext} piece: {@link Kind#TEXT} the family's marker text, {@link Kind#IMAGE_ID}
 * the MiniCPM-V image-id tag, {@link Kind#ROWS} rows at one position each, {@link Kind#GRID} rows on an
 * {@code nx * ny} token grid (with the retention mask of a pruned video frame), {@link Kind#LEAD_PADDED} a block
 * whose first rows are alignment pads, and {@link Kind#CANVAS} a MiniCPM-V canvas of an overview and a tile grid.
 * The value is what an {@link EncoderCacheConnector} transfers between servers, so a server that holds it needs
 * neither the media bytes nor the encoder to append the item. Immutable; the row arrays are held as given.
 */
public final class EncoderCacheEntry {

    /** What one chunk of an item's encoder output is. */
    public enum Kind {
        /** Marker text. */
        TEXT,
        /** The MiniCPM-V image-id tag. */
        IMAGE_ID,
        /** Rows at one position each. */
        ROWS,
        /** Rows on an {@code nx * ny} token grid, optionally pruned by a retention mask. */
        GRID,
        /** A block of rows whose first {@code lead} rows are alignment pads. */
        LEAD_PADDED,
        /** A MiniCPM-V canvas: an overview and a {@code nx x ny} tile grid. */
        CANVAS
    }

    /** One canvas tile: a {@code tokensX x tokensY} token grid followed by {@code separators} rows. */
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

        /** Rows after the grid. */
        public int separators() {
            return separators;
        }

        /** Tokens of the tile, separator rows included. */
        public int rows() {
            return tokensX * tokensY + separators;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof CanvasTile)) {
                return false;
            }
            CanvasTile other = (CanvasTile) o;
            return tokensX == other.tokensX && tokensY == other.tokensY && separators == other.separators;
        }

        @Override
        public int hashCode() {
            return Objects.hash(tokensX, tokensY, separators);
        }

        @Override
        public String toString() {
            return tokensX + "x" + tokensY + "+" + separators;
        }
    }

    /** One piece of an item's encoder output. */
    public static final class Chunk {
        private final Kind kind;
        private final String text;
        private final float[][] rows;
        private final int nx;
        private final int ny;
        private final int lead;
        private final List<CanvasTile> canvas;
        private final boolean[] mask;

        private Chunk(Kind kind, String text, float[][] rows, int nx, int ny, int lead, List<CanvasTile> canvas,
                      boolean[] mask) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.text = text;
            this.rows = rows;
            this.nx = nx;
            this.ny = ny;
            this.lead = lead;
            this.canvas = canvas;
            this.mask = mask;
        }

        /** A marker-text chunk; an empty text is the piece-free marker an empty marker text encodes to. */
        public static Chunk text(String text) {
            return new Chunk(Kind.TEXT, Objects.requireNonNull(text, "text"), null, 0, 0, 0, null, null);
        }

        /** The MiniCPM-V image-id tag. */
        public static Chunk imageId() {
            return new Chunk(Kind.IMAGE_ID, null, null, 0, 0, 0, null, null);
        }

        /** Rows at one position each. */
        public static Chunk rows(float[][] rows) {
            requireRows(rows);
            return new Chunk(Kind.ROWS, null, rows, rows.length, 1, 0, null, null);
        }

        /** Rows on an {@code nx * ny} token grid. */
        public static Chunk grid(float[][] rows, int nx, int ny) {
            return grid(rows, nx, ny, null);
        }

        /**
         * Rows on an {@code nx * ny} token grid, the kept cells' rows of a pruned grid in raster order.
         *
         * @param mask the retention mask of the grid, or null; its length is {@code nx * ny}
         * @throws IllegalArgumentException when the mask length is not the grid's, or the rows are more than the
         *                                  kept cells
         */
        public static Chunk grid(float[][] rows, int nx, int ny, boolean[] mask) {
            requireRows(rows);
            if (nx <= 0 || ny <= 0) {
                throw new IllegalArgumentException("a grid chunk needs a positive token grid, got " + nx + "x" + ny);
            }
            if (mask != null) {
                if (mask.length != nx * ny) {
                    throw new IllegalArgumentException("a " + nx + "x" + ny + " grid takes a mask of " + nx * ny
                            + " cells, got " + mask.length);
                }
                int kept = 0;
                for (boolean cell : mask) {
                    if (cell) {
                        kept++;
                    }
                }
                if (rows.length > kept) {
                    throw new IllegalArgumentException("a pruned " + nx + "x" + ny + " grid keeps " + kept
                            + " rows, the chunk holds " + rows.length);
                }
            }
            return new Chunk(Kind.GRID, null, rows, nx, ny, 0, null, mask);
        }

        /** A block of rows whose first {@code lead} rows are pads the prompt may drop. */
        public static Chunk leadPadded(float[][] rows, int lead) {
            requireRows(rows);
            if (lead < 0 || lead >= rows.length) {
                throw new IllegalArgumentException("a block of " + rows.length + " rows cannot open with " + lead
                        + " lead pads");
            }
            return new Chunk(Kind.LEAD_PADDED, null, rows, rows.length, 1, lead, null, null);
        }

        /** The rows of a canvas chunk: the overview and a {@code gridX x gridY} slice grid of {@code tiles}. */
        public static Chunk canvas(float[][] rows, int gridX, int gridY, List<CanvasTile> tiles) {
            requireRows(rows);
            if (gridX <= 0 || gridY <= 0) {
                throw new IllegalArgumentException("a canvas chunk needs a positive slice grid, got " + gridX
                        + "x" + gridY);
            }
            Objects.requireNonNull(tiles, "tiles");
            List<CanvasTile> copy = new ArrayList<>(tiles.size());
            int total = 0;
            for (CanvasTile tile : tiles) {
                copy.add(Objects.requireNonNull(tile, "canvas tile"));
                total += tile.rows();
            }
            if (total != rows.length) {
                throw new IllegalArgumentException("a canvas of " + copy.size() + " tiles holds " + total
                        + " rows, the chunk holds " + rows.length);
            }
            return new Chunk(Kind.CANVAS, null, rows, gridX, gridY, 0, Collections.unmodifiableList(copy), null);
        }

        /** What this chunk is. */
        public Kind kind() {
            return kind;
        }

        /** The marker text of a {@link Kind#TEXT} chunk, else null. */
        public String text() {
            return text;
        }

        /** The rows of a rows, grid, lead-padded or canvas chunk, else null. */
        public float[][] rows() {
            return rows;
        }

        /** Tokens per row of a {@link Kind#GRID} or {@link Kind#CANVAS} chunk, else 0. */
        public int nx() {
            return nx;
        }

        /** Token rows of a {@link Kind#GRID} or {@link Kind#CANVAS} chunk, else 0. */
        public int ny() {
            return ny;
        }

        /** The lead pads of a {@link Kind#LEAD_PADDED} chunk, else 0. */
        public int lead() {
            return lead;
        }

        /** The tiles of a {@link Kind#CANVAS} chunk, else an empty list. */
        public List<CanvasTile> canvas() {
            return canvas == null ? Collections.<CanvasTile>emptyList() : canvas;
        }

        /** The retention mask of a pruned {@link Kind#GRID} chunk, else null; the caller must not modify it. */
        public boolean[] mask() {
            return mask;
        }

        /** The rows this chunk holds. */
        public int rowCount() {
            return rows == null ? 0 : rows.length;
        }

        /** The embedding width of this chunk's rows, or 0 for a chunk without rows. */
        public int width() {
            return rows == null || rows.length == 0 ? 0 : rows[0].length;
        }

        @Override
        public String toString() {
            return kind + (text != null ? "('" + text + "')" : "") + (rows != null
                    ? "[" + rows.length + "x" + width() + (nx > 0 ? " " + nx + "x" + ny : "") + "]" : "");
        }
    }

    private final List<Chunk> chunks;
    private final int rows;
    private final int width;

    private EncoderCacheEntry(List<Chunk> chunks) {
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("an encoder cache entry needs at least one chunk");
        }
        int total = 0;
        int w = 0;
        for (int i = 0; i < chunks.size(); i++) {
            Chunk chunk = Objects.requireNonNull(chunks.get(i), "chunk " + i);
            if (chunk.rows == null) {
                continue;
            }
            if (chunk.rows.length == 0 || chunk.rows[0] == null || chunk.rows[0].length == 0) {
                throw new IllegalArgumentException("chunk " + i + " holds no embedding row");
            }
            int chunkWidth = chunk.rows[0].length;
            for (float[] row : chunk.rows) {
                if (row == null || row.length != chunkWidth) {
                    throw new IllegalArgumentException("chunk " + i + " holds rows of different widths");
                }
            }
            if (w == 0) {
                w = chunkWidth;
            } else if (chunkWidth != w) {
                throw new IllegalArgumentException("chunk " + i + " is " + chunkWidth + " values wide, the entry"
                        + " holds " + w + "-value rows");
            }
            total += chunk.rows.length;
        }
        this.chunks = Collections.unmodifiableList(new ArrayList<>(chunks));
        this.rows = total;
        this.width = w;
    }

    /**
     * An entry of {@code chunks}, in prompt order.
     *
     * @throws IllegalArgumentException when there are no chunks, a rows chunk is empty or its rows differ in width,
     *                                  or the entry's chunks differ in row width
     */
    public static EncoderCacheEntry of(List<Chunk> chunks) {
        Objects.requireNonNull(chunks, "chunks");
        return new EncoderCacheEntry(chunks);
    }

    /** The chunks, in prompt order (unmodifiable). */
    public List<Chunk> chunks() {
        return chunks;
    }

    /** The embedding rows the entry holds. */
    public int rows() {
        return rows;
    }

    /** The width of the entry's embedding rows, or 0 for an entry without rows. */
    public int width() {
        return width;
    }

    /** The bytes the entry's rows hold. */
    public long bytes() {
        long floats = 0;
        for (Chunk chunk : chunks) {
            if (chunk.rows != null) {
                floats += (long) chunk.rows.length * chunk.width();
            }
        }
        return floats * Float.BYTES;
    }

    @Override
    public String toString() {
        return "EncoderCacheEntry{chunks=" + chunks + ", rows=" + rows + ", width=" + width + "}";
    }

    private static void requireRows(float[][] rows) {
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0) {
            throw new IllegalArgumentException("an encoder cache chunk holds at least one embedding row");
        }
        if (rows[0] == null || rows[0].length == 0) {
            throw new IllegalArgumentException("an encoder cache chunk row holds at least one value");
        }
    }
}
