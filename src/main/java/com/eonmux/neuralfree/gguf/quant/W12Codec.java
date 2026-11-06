package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * The lossless 12-bit streaming codec of {@link GGMLType#W12}: one resident record per weight row, after
 * SpecPilot's w12 target weights (arXiv 2611.02501). A row of {@code cols} bfloat16-representable weights
 * ({@code cols} a whole number of 64) holds
 * <ul>
 *   <li>the Q region, one byte per weight in element order: bits 7..4 the weight's 4-bit offset {@code c} into
 *       the row's 16-binade exponent window {@code [E - 15, E]}, bit 3 its sign, bits 2..0 mantissa bits
 *       6..4;</li>
 *   <li>the B region, {@code cols / 2} bytes: mantissa bits 3..0, element {@code 2j} in the low nibble of byte
 *       {@code j}, element {@code 2j + 1} in the high nibble;</li>
 *   <li>{@value #METADATA_BYTES} bytes of row metadata: the scale {@code 2^(E-127)} as f32 (bits {@code E << 23}),
 *       the row's exact escape count {@code rp} as i32 and its Q escape count {@code rpq} as i32. Both counts are
 *       exclusive ends: an escape-table entry belongs to the row whose start count (the previous row's end, 0 for
 *       row 0) it follows.</li>
 * </ul>
 *
 * <p>The window top {@code E} is the highest exponent whose 16-binade window covers the most normal weights of
 * the row. An out-of-window weight (its binade outside the window, zero, subnormal, infinity or NaN) keeps the
 * all-zero plane code and gains an entry in one of the two escape tables that follow a shard's row region: the
 * exact table ({@value #ESCAPE_ENTRY_BYTES} bytes per entry, {@code [col i32][bf16 bits u16][pad u16][val f32]})
 * at byte {@code rows * rowBytes(cols)}, and the Q table ({@value #Q_ESCAPE_ENTRY_BYTES} bytes per entry,
 * {@code [col i32][val f32]}) for the escapes above the window ({@code |val| >= 2}) at byte
 * {@code rows * rowBytes(cols) + 12 * nEsc}. The row kernels decode every plane code with the exponent field
 * pinned at {@code 112 + c}, so the baseline value is the weight's true value times {@code 2^(127-E)} and the
 * all-zero code reads as exactly {@code 2^-15} (the Q pass's midpoint nibble: {@code 1.0625 * 2^-15}) whatever
 * the row's window; the per-entry correction {@code acc += x * val - x * 2^-15} over the exact table then
 * cancels that placeholder and adds the scaled true weight {@code val = w * 2^(127-E)} (checked exact in fp32 at
 * encode), and the epilogue scales the row sum by {@code 2^(E-127)} once — a power-of-two scaling preserves fp32
 * rounding, so a row without escapes reconstructs its BF16 matvec bit for bit. The Q kernels correct only the
 * Q-table entries, over the midpoint placeholder; the escapes below the window keep their placeholder, an error
 * far inside the 8-bit view's half-binade bound.</p>
 *
 * <p>The Q region alone is the nested 8-bit view of the weights: replacing each B nibble with the fixed midpoint
 * {@code 0x8} rounds an in-band weight to its window binade's sign/exponent/top-3-mantissa code (at most 1/16
 * relative error). The W12 Q-view kernels stream only the Q regions and read the view in registers for the fast
 * verification pass of speculative decoding.</p>
 *
 * <p>A row occupies {@link #rowBytes} {@code = 96 * (cols / 64) + 12} bytes of planes plus metadata
 * ({@link GGMLType#W12}'s 96-byte block footprint per 64 elements), so a whole record of {@code rows} weights
 * needs {@code rows * rowBytes(cols) + 12 * nEsc + 8 * nEscQ} bytes. The record is a device-resident form only;
 * no GGUF file ever stores it.</p>
 *
 * <p>Stateless and thread-safe. All methods read and write little-endian bytes by absolute index, independent of
 * the buffers' byte order.</p>
 */
public final class W12Codec {

    /** Bytes of one exact escape table entry ({@code col}, the bf16 bits, 2 pad, the scaled true weight). */
    public static final int ESCAPE_ENTRY_BYTES = 12;
    /** Bytes of one row's metadata: the scale f32, the exact escape count i32, the Q escape count i32. */
    public static final int METADATA_BYTES = 12;
    /** Bytes of one Q escape table entry ({@code col}, the scaled true weight). */
    public static final int Q_ESCAPE_ENTRY_BYTES = 8;
    /** Elements per block footprint ({@link GGMLType#W12}). */
    public static final int BLOCK_ELEMENTS = GGMLType.W12.getBlockSize();
    /** Plane bytes of one block footprint ({@link GGMLType#W12}); a row's planes are a whole number of them. */
    public static final int BLOCK_BYTES = GGMLType.W12.getTypeSize();
    /** Fixed midpoint nibble the 8-bit Q view reads the B region as. */
    public static final int QVIEW_NIBBLE = 0x8;

    /** Smallest and largest window top {@code E} (bf16 exponent field 1..254; the window is {@code [E-15, E]}).
     * The top stops at 254 so the scale stays a finite power of two for the kernels' scale multiply; weights
     * with the special exponent 255 (infinity, NaN) escape in every window. */
    private static final int MIN_WINDOW_TOP = 15;
    private static final int MAX_WINDOW_TOP = 254;
    /** The window top of a row without in-band weights; its scale is 1.0 and every weight escapes. */
    private static final int EMPTY_WINDOW_TOP = 127;
    /** bf16 exponent field of infinities and NaNs. */
    private static final int EXPONENT_SPECIAL = 0xFF;
    /** bf16 exponent field the row kernels pin the plane codes to: {@code 2^(112-127) = 2^-15} baseline. */
    static final int PLANE_EXPONENT_BASE = 112;
    /** The plane placeholder of an escaped weight: the all-zero code reads as 2^-15. */
    static final float PLACEHOLDER = (float) Math.pow(2.0, PLANE_EXPONENT_BASE - 127);
    /** The Q pass's placeholder over the midpoint nibble. */
    static final float Q_PLACEHOLDER = 1.0625f * PLACEHOLDER;
    /** A Q-table escape: its scaled true weight is at least this large, i.e. above the row window. */
    private static final float Q_ESCAPE_MIN_ABS = 2.0f;

    private W12Codec() {
    }

    // ------------------------------------------------------------------ layout

    /**
     * Plane bytes of one row: {@code cols / 64} block footprints of 96 bytes.
     *
     * @throws IllegalArgumentException when {@code cols} is not a positive multiple of {@link #BLOCK_ELEMENTS}
     */
    public static int planeBytes(int cols) {
        checkCols(cols);
        return cols / BLOCK_ELEMENTS * BLOCK_BYTES;
    }

    /**
     * Plane-plus-metadata bytes of one row of width {@code cols}: {@link #planeBytes} plus the scale and the
     * two escape counts.
     *
     * @throws IllegalArgumentException when {@code cols} is not a positive multiple of {@link #BLOCK_ELEMENTS}
     */
    public static int rowBytes(int cols) {
        checkCols(cols);
        return planeBytes(cols) + METADATA_BYTES;
    }

    private static void checkCols(int cols) {
        if (cols <= 0 || cols % BLOCK_ELEMENTS != 0) {
            throw new IllegalArgumentException(cols + " columns are not a positive multiple of the "
                    + BLOCK_ELEMENTS + "-element " + GGMLType.W12 + " block footprint");
        }
    }

    private static void checkRows(int rows) {
        if (rows <= 0) {
            throw new IllegalArgumentException(rows + " rows are not a positive row count");
        }
    }

    // ------------------------------------------------------------------ encoding

    /**
     * Scans one row of {@code cols} bf16-representable weights at {@code values[valueOff ...]}: the row's window
     * top, plane codes and escape entries. The returned plan writes itself into a record with
     * {@link RowPlan#writeRow}.
     *
     * @param row the row's index in its tensor, for the refusal messages only
     * @throws IllegalArgumentException naming {@code tensor}, the row and the element when a value is not
     *                               bf16-representable, or when its scaled true weight {@code w * 2^(127-E)} is
     *                               not exact in fp32 (a weight roughly 250 binades away from the row's window,
     *                               where the exact fp32 correction the format promises cannot be stored);
     *                               naming {@code tensor} for a bad row width or range
     */
    public static RowPlan scanRow(String tensor, float[] values, int valueOff, int cols, int row) {
        Objects.requireNonNull(tensor, "tensor");
        checkCols(cols);
        if (valueOff < 0 || valueOff + (long) cols > values.length) {
            throw new IllegalArgumentException(tensor + ": row " + row + " leaves the values");
        }
        int[] hist = new int[EXPONENT_SPECIAL + 1];
        int[] bf = new int[cols];
        for (int k = 0; k < cols; k++) {
            bf[k] = bf16Bits(tensor, values[valueOff + k], row, k);
            int e = (bf[k] >> 7) & 0xFF;
            if (e >= 1 && e <= 254) {
                hist[e]++;
            }
        }
        int top = windowTop(hist);
        byte[] qBytes = new byte[cols];
        byte[] nibbles = new byte[cols];
        int[] escCols = new int[cols];
        int[] escBf16 = new int[cols];
        float[] escVals = new float[cols];
        int[] qEscCols = new int[cols];
        float[] qEscVals = new float[cols];
        int escapes = 0;
        int qEscapes = 0;
        for (int k = 0; k < cols; k++) {
            int code = inBand(bf[k], top);
            if (code >= 0) {
                qBytes[k] = (byte) (code >>> 4);
                nibbles[k] = (byte) (code & 0xF);
                continue;
            }
            int valBits = scaledValueBits(tensor, bf[k], top, row, k);
            float val = Float.intBitsToFloat(valBits);
            escCols[escapes] = k;
            escBf16[escapes] = bf[k];
            escVals[escapes] = val;
            escapes++;
            if (Math.abs(val) >= Q_ESCAPE_MIN_ABS) {
                qEscCols[qEscapes] = k;
                qEscVals[qEscapes] = val;
                qEscapes++;
            }
        }
        return new RowPlan(top, cols, qBytes, nibbles, Arrays.copyOf(escCols, escapes),
                Arrays.copyOf(escBf16, escapes), Arrays.copyOf(escVals, escapes),
                Arrays.copyOf(qEscCols, qEscapes), Arrays.copyOf(qEscVals, qEscapes));
    }

    /** The bf16 bits of {@code v}, refusing a value whose f32 mantissa keeps bits the bf16 value would drop.
     * The raw bits keep a NaN's sign and payload, which {@code floatToIntBits} would canonicalize away. */
    private static int bf16Bits(String tensor, float v, int row, int k) {
        int bits = Float.floatToRawIntBits(v);
        if (!isOnBf16Grid(v)) {
            throw new IllegalArgumentException(tensor + ": row " + row + " element " + k
                    + " is not a bfloat16 value (f32 bits 0x" + Integer.toHexString(bits) + "); " + GGMLType.W12
                    + " stores bfloat16-representable weights losslessly only");
        }
        return bits >>> 16;
    }

    /** Whether {@code v} lies on the bfloat16 grid: its f32 mantissa keeps no bits the bf16 value would drop
     * (infinities and NaNs, whose exponent is special, always do). */
    public static boolean isOnBf16Grid(float v) {
        int bits = Float.floatToRawIntBits(v);
        return (bits & 0xFFFF) == 0 || (bits & 0x7F800000) == 0x7F800000;
    }

    /**
     * The f32 bits of the bfloat16 value nearest {@code v}: round-to-nearest-even, the grid every weight a
     * {@link GGMLType#W12} record stores lies on. A value already on the grid keeps its bits; the largest
     * finite value rounds to infinity, as round-to-nearest-even does; infinities and NaNs (the special
     * exponent, which {@link #bf16Bits} accepts) keep their bits' top half.
     */
    public static int bf16GridBits(float v) {
        int bits = Float.floatToRawIntBits(v);
        if (((bits >>> 23) & 0xFF) == EXPONENT_SPECIAL) {
            return bits & 0xFFFF0000;
        }
        return (bits + 0x7FFF + ((bits >>> 16) & 1)) & 0xFFFF0000;
    }

    /** The bfloat16 value nearest {@code v} (round-to-nearest-even): the weight {@link #bf16Bits} stores. */
    public static float bf16Grid(float v) {
        return Float.intBitsToFloat(bf16GridBits(v));
    }

    /** The row's window top: the highest exponent whose 16-binade window {@code [E-15, E]} covers the most
     * normal weights (ties to the highest); {@code 127} (scale 1.0) for a row without any. */
    private static int windowTop(int[] hist) {
        int best = EMPTY_WINDOW_TOP;
        int bestCover = 0;
        for (int top = MIN_WINDOW_TOP; top <= MAX_WINDOW_TOP; top++) {
            int cover = 0;
            for (int e = Math.max(1, top - 15); e <= Math.min(254, top); e++) {
                cover += hist[e];
            }
            if (cover >= bestCover && cover > 0) {
                bestCover = cover;
                best = top;
            }
        }
        return best;
    }

    /**
     * The 12-bit code of the bf16 number {@code bf} under window top {@code E}: bits 15..8 the Q byte
     * ({@code (c << 4) | (s << 3) | m654}), bits 3..0 the B nibble; -1 when the weight must escape (its exponent
     * is 0 or special, or outside the window). The all-zero code is a valid in-band code (the window's smallest
     * value, {@code +2^(E-15)}); an escaped weight keeps the all-zero code bytes, and the escape tables, not the
     * code bytes, name it.
     */
    private static int inBand(int bf, int top) {
        int e = (bf >> 7) & 0xFF;
        if (e == 0 || e == EXPONENT_SPECIAL) {
            return -1;
        }
        int c = e - (top - 15);
        if (c < 0 || c > 15) {
            return -1;
        }
        int q = (c << 4) | ((bf >>> 12) & 0x8) | ((bf >> 4) & 0x7);
        return (q << 4) | (bf & 0xF);
    }

    /**
     * The scaled true weight of an escaped weight as f32 bits: {@code w * 2^(127-E)}, exact in fp32.
     *
     * <p>The double product is itself exact for every bf16 value and window top (the scaled value stays inside
     * {@code 2^-260 .. 2^240}, far from both double limits), so the f32 rounding of it is exact exactly when the
     * value is representable in f32 — a power-of-two multiple of an 8-bit-significand bf16 value fits f32's
     * 24-bit significand at any exponent, and the check refuses only when the f32 exponent range would clip it
     * (to an infinity or a flushed subnormal).</p>
     *
     * @throws IllegalArgumentException naming {@code tensor}, the row and the element when the scaled true
     *                               weight is not exact in fp32
     */
    private static int scaledValueBits(String tensor, int bf, int top, int row, int k) {
        int e = (bf >> 7) & 0xFF;
        if (e == EXPONENT_SPECIAL) {
            return bf << 16;
        }
        int sig;
        int exp;
        if (e >= 1) {
            sig = 0x80 | (bf & 0x7F);
            exp = e - top - 7;
        } else {
            sig = bf & 0x7F;
            exp = -top - 6;
        }
        double scaled = Math.scalb((double) sig, exp);
        float f = (float) scaled;
        if ((double) f != scaled) {
            throw new IllegalArgumentException(tensor + ": row " + row + " element " + k + " escapes its row's"
                    + " window top " + top + ", and its scaled true weight " + scaled + " (= w * 2^(127-" + top
                    + ")) is not exact in fp32; " + GGMLType.W12 + " needs the exact fp32 correction, so leave"
                    + " the tensor in its stored type");
        }
        // the scale is a positive power of two, so the scaled value keeps the weight's sign
        return Float.floatToRawIntBits((bf & 0x8000) != 0 ? -f : f);
    }

    /** An immutable scanned row: its window top, plane codes and escape entries, writable into a record. */
    public static final class RowPlan {

        private final int top;
        private final int cols;
        private final byte[] qBytes;
        private final byte[] nibbles;
        private final int[] escCols;
        private final int[] escBf16;
        private final float[] escVals;
        private final int[] qEscCols;
        private final float[] qEscVals;

        private RowPlan(int top, int cols, byte[] qBytes, byte[] nibbles, int[] escCols, int[] escBf16,
                        float[] escVals, int[] qEscCols, float[] qEscVals) {
            this.top = top;
            this.cols = cols;
            this.qBytes = qBytes;
            this.nibbles = nibbles;
            this.escCols = escCols;
            this.escBf16 = escBf16;
            this.escVals = escVals;
            this.qEscCols = qEscCols;
            this.qEscVals = qEscVals;
        }

        /** The row's window top {@code E}. */
        public int top() {
            return top;
        }

        /** Exact escape entries of the row. */
        public int nEscapes() {
            return escCols.length;
        }

        /** Q escape entries of the row (the escapes above the window). */
        public int nQEscapes() {
            return qEscCols.length;
        }

        /**
         * Writes the row into {@code dst} at byte {@code rowBase} (planes, then scale and escape counts), and
         * appends its exact escape entries into {@code escapes} at byte {@code escapePos} and its Q escape
         * entries into {@code qEscapes} at byte {@code qEscapePos}. The counts stored in the row metadata are
         * the entry counts those positions hold after this row, so the caller advances each position by
         * {@code 12 * nEscapes()} / {@code 8 * nQEscapes()} bytes between rows of one shard. The positions must
         * hold whole entries, and the four buffers must cover the writes.
         */
        public void writeRow(ByteBuffer dst, int rowBase, ByteBuffer escapes, int escapePos,
                             ByteBuffer qEscapes, int qEscapePos) {
            int escBefore = entries(escapePos, ESCAPE_ENTRY_BYTES, "escape position");
            int qEscBefore = entries(qEscapePos, Q_ESCAPE_ENTRY_BYTES, "Q escape position");
            int planes = planeBytes(cols);
            for (int k = 0; k < cols; k++) {
                dst.put(rowBase + k, qBytes[k]);
            }
            for (int j = 0; j < cols / 2; j++) {
                dst.put(rowBase + cols + j, (byte) (nibbles[2 * j] | (nibbles[2 * j + 1] << 4)));
            }
            writeU32Le(dst, rowBase + planes, top << 23);
            writeU32Le(dst, rowBase + planes + 4, escBefore + escCols.length);
            writeU32Le(dst, rowBase + planes + 8, qEscBefore + qEscCols.length);
            for (int i = 0; i < escCols.length; i++) {
                int at = escapePos + i * ESCAPE_ENTRY_BYTES;
                writeU32Le(escapes, at, escCols[i]);
                writeU16Le(escapes, at + 4, escBf16[i]);
                writeU16Le(escapes, at + 6, 0);
                writeU32Le(escapes, at + 8, Float.floatToRawIntBits(escVals[i]));
            }
            for (int i = 0; i < qEscCols.length; i++) {
                int at = qEscapePos + i * Q_ESCAPE_ENTRY_BYTES;
                writeU32Le(qEscapes, at, qEscCols[i]);
                writeU32Le(qEscapes, at + 4, Float.floatToRawIntBits(qEscVals[i]));
            }
        }

        private static int entries(int pos, int entryBytes, String what) {
            if (pos < 0 || pos % entryBytes != 0) {
                throw new IllegalArgumentException(what + " " + pos + " is not a whole number of " + entryBytes
                        + "-byte entries");
            }
            return pos / entryBytes;
        }
    }

    /**
     * Encodes a whole record of {@code rows} rows of {@code cols} bf16-representable weights (row-major in
     * {@code values}) into a new array: the row region, then the exact escape table, then the Q escape table.
     * Rows are scanned twice — a count pass fixes the table sizes before the array is allocated, a fill pass
     * writes — so the record carries no capacity slack.
     *
     * @throws IllegalArgumentException as {@link #scanRow}, or when the record or its tables leave the
     *                               indexable range
     */
    public static byte[] encode(String tensor, float[] values, int rows, int cols) {
        Objects.requireNonNull(tensor, "tensor");
        checkCols(cols);
        checkRows(rows);
        if ((long) rows * cols > values.length) {
            throw new IllegalArgumentException(tensor + ": the values hold " + values.length + " elements, its "
                    + rows + " rows of " + cols + " need " + (long) rows * cols);
        }
        int stride = rowBytes(cols);
        if ((long) rows * stride > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException(tensor + ": " + rows + " rows of " + cols + " leave the"
                    + " indexable range");
        }
        long nEsc = 0;
        long nEscQ = 0;
        for (int r = 0; r < rows; r++) {
            RowPlan plan = scanRow(tensor, values, r * cols, cols, r);
            nEsc += plan.nEscapes();
            nEscQ += plan.nQEscapes();
            if (nEsc > (Integer.MAX_VALUE - 8L) / ESCAPE_ENTRY_BYTES
                    || nEscQ > (Integer.MAX_VALUE - 8L) / Q_ESCAPE_ENTRY_BYTES) {
                throw new IllegalArgumentException(tensor + ": row " + r + " pushes the record's escape tables"
                        + " past the indexable range");
            }
        }
        byte[] record = new byte[rows * stride + (int) nEsc * ESCAPE_ENTRY_BYTES
                + (int) nEscQ * Q_ESCAPE_ENTRY_BYTES];
        ByteBuffer out = ByteBuffer.wrap(record);
        fill(out, tensor, values, rows, cols, nEsc, nEscQ);
        return record;
    }

    /** The fill pass of {@link #encode} over an allocated record: rows in order, each appending its escape
     * entries at the running table positions, which {@link RowPlan#writeRow} takes table-relative. */
    private static void fill(ByteBuffer out, String tensor, float[] values, int rows, int cols,
                             long nEsc, long nEscQ) {
        int stride = rowBytes(cols);
        int escBase = rows * stride;
        int qEscBase = escBase + (int) nEsc * ESCAPE_ENTRY_BYTES;
        ByteBuffer escTable = out.duplicate().position(escBase).limit(out.capacity() - (int) nEscQ
                * Q_ESCAPE_ENTRY_BYTES).slice();
        ByteBuffer qTable = out.duplicate().position(qEscBase).limit(out.capacity()).slice();
        int escapePos = 0;
        int qEscapePos = 0;
        for (int r = 0; r < rows; r++) {
            RowPlan plan = scanRow(tensor, values, r * cols, cols, r);
            plan.writeRow(out, r * stride, escTable, escapePos, qTable, qEscapePos);
            escapePos += plan.nEscapes() * ESCAPE_ENTRY_BYTES;
            qEscapePos += plan.nQEscapes() * Q_ESCAPE_ENTRY_BYTES;
        }
    }

    // ------------------------------------------------------------------ decoding

    /**
     * Decodes a whole record ({@link #encode} layout) of {@code rows} rows of {@code cols} weights from
     * {@code src} into {@code dst[row * cols + k]} as F32. Every output element equals the encoded bf16 value
     * exactly (widening a bf16 number to f32 keeps its bits); an escaped weight's entry overrides its plane
     * code, whose all-zero bytes name no value of their own.
     */
    public static void decode(ByteBuffer src, float[] dst, int rows, int cols) {
        checkCols(cols);
        checkRows(rows);
        int stride = rowBytes(cols);
        int escBase = rows * stride;
        int rp = 0;
        for (int r = 0; r < rows; r++) {
            int base = r * stride;
            int planes = planeBytes(cols);
            int top = scaleTop(src, base, planes);
            int end = readU32Le(src, base + planes + 4);
            if (end < rp) {
                throw new IllegalArgumentException("row " + r + " ends its escapes at " + end
                        + ", before the previous row's end " + rp);
            }
            for (int k = 0; k < cols; k++) {
                dst[r * cols + k] = Float.intBitsToFloat(planeBf16Bits(src, base, cols, k, top, false) << 16);
            }
            for (int i = rp; i < end; i++) {
                int at = escBase + i * ESCAPE_ENTRY_BYTES;
                int col = readU32Le(src, at);
                checkEscapeCol(col, cols, r, i);
                dst[r * cols + col] = Float.intBitsToFloat(readU16Le(src, at + 4) << 16);
            }
            rp = end;
        }
    }

    /**
     * The exact value of element {@code col} of row {@code row}: its plane code, overridden by its escape-table
     * entry when the weight escapes.
     */
    public static float element(ByteBuffer src, int rows, int cols, int row, int col) {
        int colsOk = checkElementArgs(rows, cols, row, col);
        int planes = planeBytes(colsOk);
        int stride = rowBytes(colsOk);
        int base = row * stride;
        int escBase = rows * stride;
        int start = row == 0 ? 0 : readU32Le(src, base - stride + planes + 4);
        int end = readU32Le(src, base + planes + 4);
        for (int i = start; i < end; i++) {
            int at = escBase + i * ESCAPE_ENTRY_BYTES;
            if (readU32Le(src, at) == col) {
                return Float.intBitsToFloat(readU16Le(src, at + 4) << 16);
            }
        }
        int top = scaleTop(src, base, planes);
        return Float.intBitsToFloat(planeBf16Bits(src, base, cols, col, top, false) << 16);
    }

    /**
     * The 8-bit Q-view value of element {@code col} of row {@code row}: the Q byte with the B nibble read as
     * {@link #QVIEW_NIBBLE}; an escape keeps the midpoint placeholder unless the Q table corrects it, in which
     * case the view carries its scaled true weight over that placeholder.
     */
    public static float qViewElement(ByteBuffer src, int rows, int cols, int row, int col) {
        int colsOk = checkElementArgs(rows, cols, row, col);
        int planes = planeBytes(colsOk);
        int stride = rowBytes(colsOk);
        int base = row * stride;
        int escBase = rows * stride;
        int top = scaleTop(src, base, planes);
        float scale = Float.intBitsToFloat(top << 23);
        int escStart = row == 0 ? 0 : readU32Le(src, base - stride + planes + 4);
        int escEnd = readU32Le(src, base + planes + 4);
        for (int i = escStart; i < escEnd; i++) {
            if (readU32Le(src, escBase + i * ESCAPE_ENTRY_BYTES) == col) {
                int qStart = row == 0 ? 0 : readU32Le(src, base - stride + planes + 8);
                int qEnd = readU32Le(src, base + planes + 8);
                int qEscBase = qTableBase(src, rows, colsOk);
                for (int q = qStart; q < qEnd; q++) {
                    int at = qEscBase + q * Q_ESCAPE_ENTRY_BYTES;
                    if (readU32Le(src, at) == col) {
                        return (Float.intBitsToFloat(readU32Le(src, at + 4)) + Q_PLACEHOLDER) * scale;
                    }
                }
                return Q_PLACEHOLDER * scale;
            }
        }
        return Float.intBitsToFloat(planeBf16Bits(src, base, cols, col, top, true) << 16);
    }

    /** The byte base of the Q escape table: after the exact table, whose size the last row's count holds. */
    private static int qTableBase(ByteBuffer src, int rows, int cols) {
        int stride = rowBytes(cols);
        int planes = planeBytes(cols);
        int escBase = rows * stride;
        int nEsc = readU32Le(src, (rows - 1) * stride + planes + 4);
        return escBase + nEsc * ESCAPE_ENTRY_BYTES;
    }

    private static int checkElementArgs(int rows, int cols, int row, int col) {
        checkCols(cols);
        checkRows(rows);
        if (row < 0 || row >= rows) {
            throw new IllegalArgumentException("row " + row + " leaves the " + rows + "-row record");
        }
        if (col < 0 || col >= cols) {
            throw new IllegalArgumentException("element " + col + " leaves the row width " + cols);
        }
        return cols;
    }

    private static void checkEscapeCol(int col, int cols, int row, int i) {
        if (col >= cols) {
            throw new IllegalArgumentException("row " + row + " escape " + i + " names element " + col
                    + ", outside the row width " + cols);
        }
    }

    /** The window top from the row's scale word at {@code base + planes}. */
    private static int scaleTop(ByteBuffer src, int base, int planes) {
        return (int) ((readU32Le(src, base + planes) >>> 23) & 0xFF);
    }

    /**
     * The true bf16 bits of element {@code k} of the row at {@code base}: its Q byte at {@code base + k}
     * and B nibble in the row's B region at {@code base + cols} (or the midpoint {@link #QVIEW_NIBBLE} for
     * the 8-bit view), exponent field {@code c + top - 15}. An escaped weight's all-zero bytes read as the
     * window's smallest value — callers override it from its escape entry.
     */
    private static int planeBf16Bits(ByteBuffer src, int base, int cols, int k, int top, boolean view) {
        int q = readU8(src, base + k);
        int bByte = readU8(src, base + cols + (k >> 1));
        int nibble = view ? QVIEW_NIBBLE : (k & 1) == 0 ? bByte & 0xF : (bByte >> 4) & 0xF;
        return decodeBf16Bits(q, nibble, top);
    }

    /** The unsigned byte at absolute byte {@code at}. */
    private static int readU8(ByteBuffer src, int at) {
        return src.get(at) & 0xFF;
    }

    /** The unsigned little-endian 16-bit value at absolute byte {@code at}. */
    private static int readU16Le(ByteBuffer src, int at) {
        return readU8(src, at) | readU8(src, at + 1) << 8;
    }

    /** The unsigned little-endian 32-bit value at absolute byte {@code at}. */
    private static int readU32Le(ByteBuffer src, int at) {
        return readU8(src, at) | readU8(src, at + 1) << 8
                | readU8(src, at + 2) << 16 | readU8(src, at + 3) << 24;
    }

    private static void writeU16Le(ByteBuffer dst, int at, int v) {
        dst.put(at, (byte) v);
        dst.put(at + 1, (byte) (v >>> 8));
    }

    private static void writeU32Le(ByteBuffer dst, int at, int v) {
        dst.put(at, (byte) v);
        dst.put(at + 1, (byte) (v >>> 8));
        dst.put(at + 2, (byte) (v >>> 16));
        dst.put(at + 3, (byte) (v >>> 24));
    }

    /** The bf16 bits of element {@code k} from its Q byte, B nibble and window top. */
    private static int decodeBf16Bits(int q, int nibble, int top) {
        int c = (q >> 4) & 0xF;
        int s = (q >> 3) & 1;
        int m654 = q & 0x7;
        int e = c + top - 15;
        return (s << 15) | (e << 7) | (m654 << 4) | nibble;
    }
}
