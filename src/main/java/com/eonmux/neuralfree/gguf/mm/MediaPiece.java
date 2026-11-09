package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * One piece of an encoded medium: marker text, an image-id tag, or a chunk of rows (one position each, a token
 * grid, a canvas, or a lead-padded block). Immutable.
 */
final class MediaPiece {
    enum Kind { TEXT, IMAGE_ID, ROWS, GRID, LEAD_PADDED, CANVAS }

    final Kind kind;
    final String text;
    final float[][] rows;
    /** The token grid of a {@link Kind#GRID} piece, the slice grid of a {@link Kind#CANVAS} piece. */
    final int nx;
    final int ny;
    /** The lead pad rows of a {@link Kind#LEAD_PADDED} piece; 0 otherwise. */
    final int lead;
    /** The tiles of a {@link Kind#CANVAS} piece; null otherwise. */
    final List<MediaRopePositions.CanvasTile> canvas;
    /**
     * The retention mask of a pruned {@link Kind#GRID} piece, one entry per grid cell in raster order, the
     * rows the kept cells'; null for an unpruned piece.
     */
    final boolean[] mask;

    private MediaPiece(Kind kind, String text, float[][] rows, int nx, int ny, int lead,
                       List<MediaRopePositions.CanvasTile> canvas, boolean[] mask) {
        this.kind = kind;
        this.text = text;
        this.rows = rows;
        this.nx = nx;
        this.ny = ny;
        this.lead = lead;
        this.canvas = canvas;
        this.mask = mask;
    }

    static MediaPiece text(String text) {
        return new MediaPiece(Kind.TEXT, text, null, 0, 0, 0, null, null);
    }

    static MediaPiece imageId() {
        return new MediaPiece(Kind.IMAGE_ID, null, null, 0, 0, 0, null, null);
    }

    /** Rows taking one position each (a slice, an overview, an audio window). */
    static MediaPiece rows(float[][] rows) {
        requireRows(rows);
        return new MediaPiece(Kind.ROWS, null, rows, rows.length, 1, 0, null, null);
    }

    /** Rows on an {@code nx * ny} token grid. */
    static MediaPiece grid(float[][] rows, int nx, int ny) {
        requireRows(rows);
        return new MediaPiece(Kind.GRID, null, rows, nx, ny, 0, null, null);
    }

    /**
     * Rows on an {@code nx * ny} token grid, the mask of a shared-cache chunk they were pruned by travelling
     * with them for their positions.
     */
    static MediaPiece grid(float[][] rows, int nx, int ny, boolean[] mask) {
        requireRows(rows);
        return new MediaPiece(Kind.GRID, null, rows, nx, ny, 0, null, mask);
    }

    /**
     * This grid chunk pruned to the kept cells of {@code mask[from .. from + cells)}: the rows become the
     * kept cells' rows in raster order, and the mask travels with the chunk for its positions.
     */
    MediaPiece withMask(boolean[] mask, int from, int cells) {
        if (kind != Kind.GRID || nx * ny != cells) {
            throw new IllegalStateException("a retention mask prunes a " + cells + "-cell grid chunk, this"
                    + " piece is a " + kind + " of " + nx * ny + " cells");
        }
        int kept = 0;
        for (int i = 0; i < cells; i++) {
            if (mask[from + i]) {
                kept++;
            }
        }
        float[][] keptRows = new float[kept][];
        int at = 0;
        for (int i = 0; i < cells; i++) {
            if (mask[from + i]) {
                keptRows[at++] = rows[i];
            }
        }
        return new MediaPiece(Kind.GRID, null, keptRows, nx, ny, 0, null,
                Arrays.copyOfRange(mask, from, from + cells));
    }

    /** A block of rows whose first {@code lead} rows are pads its place in the prompt may drop. */
    static MediaPiece leadPadded(float[][] rows, int lead) {
        requireRows(rows);
        if (lead < 0 || lead >= rows.length) {
            throw new IllegalStateException("a block of " + rows.length + " rows cannot open with " + lead
                    + " lead pads");
        }
        return new MediaPiece(Kind.LEAD_PADDED, null, rows, rows.length, 1, lead, null, null);
    }

    /**
     * The block placed after {@code tokensBefore} prompt tokens: it opens with {@code align - 1 - tokensBefore %
     * align} pads ({@code mtmd_tokenize}'s {@code lead_pad}), the trailing pads of this block's {@code lead}.
     */
    MediaPiece leadPadded(int tokensBefore, int align) {
        int keep = align - 1 - tokensBefore % align;
        if (keep > lead) {
            throw new IllegalStateException("the block carries " + lead + " lead pads, its place needs " + keep);
        }
        float[][] kept = Arrays.copyOfRange(rows, lead - keep, rows.length);
        return new MediaPiece(Kind.ROWS, null, kept, kept.length, 1, keep, null, null);
    }

    /** The rows of a canvas chunk: the overview and a {@code gridX x gridY} slice grid of {@code tiles}. */
    static MediaPiece canvas(float[][] rows, int gridX, int gridY, List<MediaRopePositions.CanvasTile> tiles) {
        requireRows(rows);
        return new MediaPiece(Kind.CANVAS, null, rows, gridX, gridY, 0,
                Collections.unmodifiableList(new ArrayList<>(tiles)), null);
    }

    private static void requireRows(float[][] rows) {
        if (rows.length == 0) {
            throw new MediaInputException("the encoder produced no embedding rows for the media");
        }
    }

    long floats() {
        return rows == null || rows.length == 0 ? 0 : (long) rows.length * rows[0].length;
    }

    /** The bytes the pieces of one encoded item hold. */
    static long bytesOf(List<MediaPiece> pieces) {
        long floats = 0;
        for (MediaPiece p : pieces) {
            floats += p.floats();
        }
        return floats * Float.BYTES;
    }
}
