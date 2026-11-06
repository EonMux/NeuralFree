package com.eonmux.neuralfree.gguf.mm;

/**
 * The separator rows an encoder appends after the image tokens of one tile (llama.cpp {@code clip_suffix_type}
 * and {@code clip_suffix_rows}, tools/mtmd/clip-impl.h): rows of the projector's {@code v.tok_embd_sep} table, the
 * text model's embeddings of {@code </image>} (row 0), {@code <slice>} (row 1), {@code </slice>} (row 2) and
 * {@code \n} (row 3) the MiniCPM-V 4.7 converter stores. The MiniCPM-V 4.7 canvas chunk closes every tile with
 * them inside the media rows instead of as text.
 */
public enum ClipSuffix {
    /** No separator rows. */
    NONE,
    /** The overview of an unsliced image: {@code </image>}. */
    MINICPMV_OV(0),
    /** The overview before the slices: {@code </image><slice>}. */
    MINICPMV_OV_SLICE(0, 1),
    /** A slice before another in its row: {@code </slice><slice>}. */
    MINICPMV_SLICE(2, 1),
    /** The last slice of a row before the next row: {@code </slice>\n<slice>}. */
    MINICPMV_ROW_END(2, 3, 1),
    /** The last slice: {@code </slice>}. */
    MINICPMV_LAST(2);

    private final int[] rows;

    ClipSuffix(int... rows) {
        this.rows = rows;
    }

    /** The {@code v.tok_embd_sep} rows appended, in order (a copy). */
    public int[] rows() {
        return rows.clone();
    }

    /** The number of appended rows. */
    public int size() {
        return rows.length;
    }
}
