package com.eonmux.neuralfree.gguf.quant;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lossless codec of {@link com.eonmux.neuralfree.gguf.GGMLType#W12}: row-record roundtrips over
 * bf16-representable values including the escapes (zeros, subnormals, special exponents, window outliers)
 * with no escape-count limit, the per-row exponent-window choice (ties to the highest), the record layout
 * (planes, metadata, exact and Q escape tables), the per-element reads, the nested 8-bit Q view, the
 * bytes-per-weight budget, and every refusal (non-bf16 values, an fp32-inexact scaled weight, malformed
 * records).
 */
public class W12CodecTest {

    private static final String T = "blk.0.attn_q.weight";

    /** The little-endian word at absolute byte {@code at} of {@code buf}. */
    private static int word(ByteBuffer buf, int at) {
        return (buf.get(at) & 0xFF) | (buf.get(at + 1) & 0xFF) << 8 | (buf.get(at + 2) & 0xFF) << 16
                | (buf.get(at + 3) & 0xFF) << 24;
    }

    /** The row's window top E: the exponent field of the scale word at its {@code planes} offset. */
    private static int windowTop(ByteBuffer record, int cols, int row) {
        int planes = W12Codec.planeBytes(cols);
        return (word(record, row * W12Codec.rowBytes(cols) + planes) >>> 23) & 0xFF;
    }

    /** The f32 with bf16 bits {@code bits}. */
    private static float bf16(int bits) {
        return Float.intBitsToFloat(bits << 16);
    }

    // ------------------------------------------------------------------ roundtrip

    @Test
    public void roundtripsRowsWithZerosDenormalsAndSpecialsBitExactly() {
        // four rows of 128: dense binades around a different center each, every row carrying zeros,
        // subnormals, infinities, NaNs and out-of-window outliers on both sides; nothing refuses
        float[] specials = {bf16(0x0000), bf16(0x8000), bf16(0x0003), bf16(0x8058),
                bf16(0x7F80), bf16(0xFF80), bf16(0x7FC0), bf16(0xFFC0)};
        int[] centers = {20, 127, 200, 238};
        int cols = 128;
        int rows = 4;
        float[] values = new float[rows * cols];
        Random r = new Random(11);
        for (int row = 0; row < rows; row++) {
            for (int k = 0; k < cols - 4; k++) {
                int sign = r.nextBoolean() ? 0x8000 : 0;
                int e = centers[row] + r.nextInt(16);
                if (k % 13 == 0) {
                    e = Math.max(1, centers[row] - 25);
                } else if (k % 17 == 0) {
                    e = Math.min(254, centers[row] + 25);
                }
                values[row * cols + k] = bf16(sign | (e << 7) | r.nextInt(128));
            }
            for (int j = 0; j < 4; j++) {
                values[row * cols + cols - 4 + j] = specials[(row * 2 + j) % specials.length];
            }
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, rows, cols));
        float[] back = new float[values.length];
        W12Codec.decode(enc, back, rows, cols);
        for (int i = 0; i < values.length; i++) {
            assertEquals("element " + i, Float.floatToRawIntBits(values[i]),
                    Float.floatToRawIntBits(back[i]));
        }
    }

    @Test
    public void anAllZeroRowRoundtripsWithTheUnitScale() {
        float[] values = new float[2 * 64];
        for (int k = 0; k < 64; k++) {
            values[64 + k] = bf16(0x8000 | (117 << 7) | ((k & 7) << 4) | (k & 15));
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 2, 64));
        assertEquals(W12Codec.planeBytes(64) + W12Codec.METADATA_BYTES, W12Codec.rowBytes(64));
        assertEquals(127, windowTop(enc, 64, 0));
        // the whole row of zeros escapes, so its planes keep the all-zero codes and all sixty-four exact
        // entries precede row 1's
        for (int k = 0; k < 64; k++) {
            assertEquals("zero code " + k, 0, enc.get(k) & 0xFF);
        }
        assertEquals(64, word(enc, W12Codec.rowBytes(64) - 8));
        assertEquals(64, word(enc, 2 * W12Codec.rowBytes(64) - 8));
        float[] back = new float[values.length];
        W12Codec.decode(enc, back, 2, 64);
        assertArrayEquals(values, back, 0.0f);
        assertEquals(0.0f, W12Codec.element(enc, 2, 64, 0, 13), 0.0f);
    }

    @Test
    public void aRowWithoutEscapesRoundtripsThroughElementReads() {
        int cols = 192;
        Random r = new Random(3);
        float[] values = new float[64 * cols];
        for (int i = 0; i < values.length; i++) {
            int sign = r.nextBoolean() ? 0x8000 : 0;
            int e = 100 + r.nextInt(16);
            values[i] = bf16(sign | (e << 7) | r.nextInt(128));
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 64, cols));
        assertEquals(0, word(enc, 64 * W12Codec.rowBytes(cols) - 8));
        Random idx = new Random(9);
        for (int probe = 0; probe < 200; probe++) {
            int row = idx.nextInt(64);
            int col = idx.nextInt(cols);
            assertEquals("element (" + row + ", " + col + ")",
                    Float.floatToRawIntBits(values[row * cols + col]),
                    Float.floatToRawIntBits(W12Codec.element(enc, 64, cols, row, col)));
        }
    }

    // ------------------------------------------------------------------ the budget

    @Test
    public void realLikeRowsStayWithinTheBytesPerWeightBudget() {
        // a decoder-FFN-like row: bf16 grid points with full 8-bit mantissas over one 16-binade window
        // (so nearly every weight codes in band), a realistic share of exact zeros and a rare far
        // outlier per few rows, everything else escaping and priced by its table entry
        int cols = 4096;
        int rows = 16;
        float[] values = new float[rows * cols];
        Random r = new Random(21);
        for (int i = 0; i < values.length; i++) {
            if (i % 2048 == 7) {
                continue; // 0.05% exact zeros, like a real projection's sparsity
            }
            if (i % 16384 == 1000) {
                values[i] = bf16((r.nextBoolean() ? 0x8000 : 0) | (153 << 7) | r.nextInt(128));
                continue; // one ~20-binade outlier every few rows
            }
            int sign = r.nextBoolean() ? 0x8000 : 0;
            int e = 118 + r.nextInt(16);
            values[i] = bf16(sign | (e << 7) | r.nextInt(128));
        }
        byte[] record = W12Codec.encode(T, values, rows, cols);
        double bitsPerWeight = record.length * 8.0 / (rows * (double) cols);
        assertTrue("bits per weight " + bitsPerWeight + " leaves the design point",
                bitsPerWeight <= 12.1);
    }

    // ------------------------------------------------------------------ layout

    @Test
    public void theRowLayoutMatchesTheDocumentedOffsets() {
        // one row of 64, no escapes: element k holds the bf16 number with sign (k & 1), exponent
        // 100 + (k % 8) and the full 7-bit mantissa (k & 7) at bits 6..4 over the nibble k & 15; the
        // tops whose 16-binade window covers all eight exponents are 107..115, and the tie goes to
        // the highest, 115
        float[] values = new float[64];
        for (int k = 0; k < 64; k++) {
            int s = (k & 1) << 15;
            int e = (100 + (k % 8)) << 7;
            int m = ((k & 7) << 4) | (k & 0xF);
            values[k] = bf16(s | e | m);
        }
        int planes = W12Codec.planeBytes(64);
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
        assertEquals(96, planes);
        assertEquals(planes + W12Codec.METADATA_BYTES, enc.limit());
        assertEquals(115, windowTop(enc, 64, 0));
        assertEquals(115 << 23, word(enc, planes));
        assertEquals(0, word(enc, planes + 4));
        assertEquals(0, word(enc, planes + 8));
        for (int k = 0; k < 64; k++) {
            int c = k % 8;
            int s = (k & 1) << 3;
            int m654 = k & 7;
            assertEquals("Q byte of element " + k, (c << 4) | s | m654, enc.get(k) & 0xFF);
            int bByte = enc.get(64 + (k >> 1)) & 0xFF;
            int nibble = (k & 1) == 0 ? bByte & 0xF : (bByte >> 4) & 0xF;
            assertEquals("B nibble of element " + k, k & 0xF, nibble);
        }
        float[] back = new float[64];
        W12Codec.decode(enc, back, 1, 64);
        assertArrayEquals(values, back, 0.0f);
    }

    @Test
    public void theEscapeTablesFollowTheRowRegionWithTheDocumentedEntries() {
        // two rows of 64: row 0 mixes one far above-window weight (a Q-table escape) with a zero, a
        // subnormal, an infinity and a NaN (exact-table-only escapes) over in-band fillers of exponent
        // 100 (the tie-to-highest top covering it is 115); row 1 is all ones (top 142)
        float[] values = new float[2 * 64];
        float zero = 0.0f;
        float sub = bf16(0x0003);
        float inf = bf16(0x7F80);
        float nan = bf16(0x7FC0);
        float above = bf16(0x3E7F);
        values[0] = zero;
        values[1] = sub;
        values[2] = inf;
        values[3] = nan;
        values[4] = above;
        for (int k = 5; k < 64; k++) {
            values[k] = bf16((100 << 7) | ((k & 7) << 4) | (k & 15));
        }
        for (int k = 0; k < 64; k++) {
            values[64 + k] = 1.0f;
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 2, 64));
        assertEquals(115, windowTop(enc, 64, 0));
        assertEquals(142, windowTop(enc, 64, 1));
        int stride = W12Codec.rowBytes(64);
        assertEquals(115 << 23, word(enc, stride - 12));
        assertEquals(5, word(enc, stride - 8));
        assertEquals(2, word(enc, stride - 4));
        assertEquals(142 << 23, word(enc, 2 * stride - 12));
        assertEquals(5, word(enc, 2 * stride - 8));
        assertEquals(2, word(enc, 2 * stride - 4));
        // the exact table at 2 * stride: row 0's escapes in element order, each with its bf16 bits and
        // its scaled true weight
        int escBase = 2 * stride;
        float[] escapes = {zero, sub, inf, nan, above};
        for (int i = 0; i < escapes.length; i++) {
            int at = escBase + i * W12Codec.ESCAPE_ENTRY_BYTES;
            assertEquals("escape " + i + " names its element", i, word(enc, at));
            assertEquals("escape " + i + " keeps the bf16 bits",
                    Float.floatToRawIntBits(escapes[i]) >>> 16, word(enc, at + 4) & 0xFFFF);
        }
        assertEquals(Float.floatToRawIntBits(above * (float) Math.pow(2.0, 127 - 115)),
                word(enc, escBase + 4 * W12Codec.ESCAPE_ENTRY_BYTES + 8));
        // the Q table follows the exact table: the infinity and the above-window escape qualify, in
        // element order, each carrying its scaled true weight
        int qBase = escBase + 5 * W12Codec.ESCAPE_ENTRY_BYTES;
        assertEquals(2, word(enc, qBase));
        assertEquals(Float.floatToRawIntBits(inf * (float) Math.pow(2.0, 127 - 115)), word(enc, qBase + 4));
        assertEquals(4, word(enc, qBase + 8));
        assertEquals(Float.floatToRawIntBits(above * (float) Math.pow(2.0, 127 - 115)), word(enc, qBase + 12));
        // the full record decodes bit-exactly across both rows
        float[] back = new float[values.length];
        W12Codec.decode(enc, back, 2, 64);
        assertArrayEquals(values, back, 0.0f);
    }

    // ------------------------------------------------------------------ window choice

    @Test
    public void windowTakesTheSmallestTopOfTheBestCover() {
        // sixty-three weights of exponent 17 and one of 33: the best cover is 63, kept by tops 17..32,
        // and the tie goes to the highest, 32; e = 33 sits outside [17, 32] and escapes
        float[] values = new float[64];
        values[0] = Float.intBitsToFloat(33 << 23);
        for (int k = 1; k < 64; k++) {
            values[k] = Float.intBitsToFloat(17 << 23);
        }
        int planes = W12Codec.planeBytes(64);
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
        assertEquals(32, windowTop(enc, 64, 0));
        assertEquals(32 << 23, word(enc, planes));
        assertEquals(1, word(enc, planes + 4));
        float[] back = new float[64];
        W12Codec.decode(enc, back, 1, 64);
        assertEquals(values[0], back[0], 0.0f);
    }

    @Test
    public void windowHoldsTheWidestSixteenBinades() {
        // sixteen consecutive exponents: exactly one window covers all of them, so the top is its own
        float[] values = new float[64];
        for (int k = 0; k < 16; k++) {
            values[k] = Float.intBitsToFloat((130 - k) << 23);
        }
        for (int k = 16; k < 64; k++) {
            values[k] = values[k % 16];
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
        assertEquals(130, windowTop(enc, 64, 0));
        assertEquals(0, word(enc, W12Codec.planeBytes(64) + 4));
        float[] back = new float[64];
        W12Codec.decode(enc, back, 1, 64);
        assertArrayEquals(values, back, 0.0f);
    }

    @Test
    public void aSingleExponentRowTakesTheTopFifteenAboveIt() {
        float[] values = new float[64];
        values[0] = Float.intBitsToFloat(120 << 23);
        for (int k = 1; k < 64; k++) {
            values[k] = 0.0f;
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
        assertEquals(135, windowTop(enc, 64, 0));
        // the 63 zeros escape into the exact table; the in-band weight does not
        assertEquals(63, word(enc, W12Codec.planeBytes(64) + 4));
        assertEquals(0, word(enc, W12Codec.planeBytes(64) + 8));
    }

    @Test
    public void eachRowTakesItsOwnWindow() {
        // row 0 lives in the binades around 2^-4, row 1 around 2^3: the windows differ and every weight
        // codes in band in its own row
        int cols = 128;
        float[] values = new float[2 * cols];
        Random r = new Random(5);
        for (int row = 0; row < 2; row++) {
            for (int k = 0; k < cols; k++) {
                int e = (row == 0 ? 118 : 134) + r.nextInt(16);
                values[row * cols + k] = bf16((r.nextBoolean() ? 0x8000 : 0) | (e << 7) | r.nextInt(128));
            }
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 2, cols));
        assertTrue(String.valueOf(windowTop(enc, cols, 0)), windowTop(enc, cols, 0) == 133);
        assertTrue(String.valueOf(windowTop(enc, cols, 1)), windowTop(enc, cols, 1) == 149);
        assertEquals(0, word(enc, W12Codec.rowBytes(cols) - 8));
        assertEquals(0, word(enc, 2 * W12Codec.rowBytes(cols) - 8));
        float[] back = new float[values.length];
        W12Codec.decode(enc, back, 2, cols);
        assertArrayEquals(values, back, 0.0f);
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void refusesValuesBf16WouldRound() {
        float[] values = new float[64];
        for (int k = 0; k < 64; k++) {
            values[k] = 1.0f;
        }
        values[7] = 0.123456789f;
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> W12Codec.encode(T, values, 1, 64));
        assertTrue(e.getMessage(), e.getMessage().contains(T));
        assertTrue(e.getMessage(), e.getMessage().contains("bfloat16"));
        assertTrue(e.getMessage(), e.getMessage().contains("element 7"));
    }

    @Test
    public void refusesASubnormalItsScaledValueCannotHoldInFp32() {
        // a row of exponent-170 fillers (top 185) with one bf16 subnormal: its scaled true weight
        // w * 2^(127 - 185) flushes below fp32's smallest subnormal, where the exact correction the
        // format promises cannot be stored, so the row refuses by tensor, row and element
        float[] values = new float[64];
        for (int k = 0; k < 64; k++) {
            values[k] = bf16((170 << 7) | ((k & 7) << 4));
        }
        values[9] = bf16(0x0040);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> W12Codec.encode(T, values, 1, 64));
        assertTrue(e.getMessage(), e.getMessage().contains(T));
        assertTrue(e.getMessage(), e.getMessage().contains("row 0"));
        assertTrue(e.getMessage(), e.getMessage().contains("element 9"));
        assertTrue(e.getMessage(), e.getMessage().contains("exact in fp32"));
    }

    @Test
    public void refusesABadRowWidthOrRange() {
        assertThrows(IllegalArgumentException.class, () -> W12Codec.encode(T, new float[100], 1, 100));
        assertThrows(IllegalArgumentException.class, () -> W12Codec.rowBytes(100));
        assertThrows(IllegalArgumentException.class, () -> W12Codec.planeBytes(-64));
        // a values array shorter than the rows need
        assertThrows(IllegalArgumentException.class, () -> W12Codec.encode(T, new float[64], 2, 64));
    }

    @Test
    public void decodeRefusesARowThatRewindsItsEscapeCounts() {
        // row 0 carries the record's only escape (one zero); rewinding row 1's end count below row 0's
        // leaves the entry claimed by no row, and the record refuses
        float[] values = new float[2 * 64];
        values[0] = 0.0f;
        for (int k = 1; k < 2 * 64; k++) {
            values[k] = 1.0f;
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 2, 64));
        int stride = W12Codec.rowBytes(64);
        assertEquals(1, word(enc, stride - 8));
        assertEquals(1, word(enc, 2 * stride - 8));
        enc.put(2 * stride - 8, (byte) 0);
        enc.put(2 * stride - 7, (byte) 0);
        enc.put(2 * stride - 6, (byte) 0);
        enc.put(2 * stride - 5, (byte) 0);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> W12Codec.decode(enc, new float[128], 2, 64));
        assertTrue(e.getMessage(), e.getMessage().contains("escape"));
    }

    @Test
    public void elementReadsRefuseOutOfRangeArguments() {
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, new float[64], 1, 64));
        assertThrows(IllegalArgumentException.class, () -> W12Codec.element(enc, 1, 64, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> W12Codec.element(enc, 1, 64, 0, 64));
        assertThrows(IllegalArgumentException.class, () -> W12Codec.qViewElement(enc, 1, 64, 0, 64));
        assertThrows(IllegalArgumentException.class, () -> W12Codec.element(enc, 1, 100, 0, 0));
    }

    // ------------------------------------------------------------------ the bf16 grid

    @Test
    public void theGridKeepsOnGridValuesAndTheSpecials() {
        int[] onGrid = {0x0000, 0x8000, 0x0080, 0x3E80, 0x3F80, 0x7F7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1};
        for (int bits : onGrid) {
            assertEquals("bf16 bits 0x" + Integer.toHexString(bits), bits,
                    W12Codec.bf16GridBits(bf16(bits)) >>> 16);
        }
        // infinities and NaNs keep their bits' top half instead of rounding a payload up the exponent
        assertEquals(0x7F800000, W12Codec.bf16GridBits(Float.intBitsToFloat(0x7F800001)));
        assertEquals(0x7FC10000, W12Codec.bf16GridBits(Float.intBitsToFloat(0x7FC12345)));
        assertEquals(0xFF800000, W12Codec.bf16GridBits(Float.intBitsToFloat(0xFF800400)));
    }

    @Test
    public void theGridRoundsToNearestEven() {
        // 1.0 + 2^-9 sits under the midpoint of [1.0, 1.0078125): down
        assertEquals(0x3F800000, W12Codec.bf16GridBits(1.001953125f));
        // 1.0 + 2^-8 is the exact midpoint; the lower neighbour's mantissa is even, so it wins
        assertEquals(0x3F800000, W12Codec.bf16GridBits(1.00390625f));
        // the midpoint above 1.0078125 (mantissa 1, odd) rounds up to mantissa 2
        assertEquals(0x3F820000, W12Codec.bf16GridBits(1.01171875f));
        // the largest finite float rounds to infinity, in both signs
        assertEquals(0x7F800000, W12Codec.bf16GridBits(Float.intBitsToFloat(0x7F7FFFFF)));
        assertEquals(0xFF800000, W12Codec.bf16GridBits(Float.intBitsToFloat(0xFF7FFFFF)));
    }

    @Test
    public void gridRoundedValuesEncodeAndDecodeBitExactly() {
        // the values an F16 tensor carries, brought onto the grid, then stored and read back
        float[] values = new float[64];
        for (int k = 0; k < 64; k++) {
            values[k] = W12Codec.bf16Grid(0.123456789f + k * 0.0005f);
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
        float[] back = new float[64];
        W12Codec.decode(enc, back, 1, 64);
        for (int k = 0; k < 64; k++) {
            assertEquals("element " + k, Float.floatToRawIntBits(values[k]),
                    Float.floatToRawIntBits(back[k]));
        }
    }

    // ------------------------------------------------------------------ the 8-bit Q view

    @Test
    public void qViewRoundsInBandElementsToTheMidpointAndKeepsItsPlaceholder() {
        float[] values = new float[64];
        // the exponent-115 and exponent-127 fillers share their widest 16-binade window [115, 130], whose
        // tie goes to the highest top 130: the exponent-100 weight, the zero and the infinity escape;
        // only the infinity's scaled value reaches the Q table
        values[0] = Float.intBitsToFloat(100 << 23);
        values[1] = Float.intBitsToFloat((1 << 31) | (115 << 23) | (7 << 20));
        values[2] = 0.0f;
        values[3] = Float.intBitsToFloat(0x7F800000);
        for (int k = 4; k < 64; k++) {
            values[k] = Float.intBitsToFloat((127 << 23) | ((k & 7) << 20));
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
        int top = windowTop(enc, 64, 0);
        assertEquals(130, top);
        int planes = W12Codec.planeBytes(64);
        assertEquals(3, word(enc, planes + 4));
        assertEquals(1, word(enc, planes + 8));
        float[] exact = new float[64];
        W12Codec.decode(enc, exact, 1, 64);
        assertArrayEquals(values, exact, 0.0f);
        float scale = Float.intBitsToFloat(top << 23);
        for (int k = 0; k < 64; k++) {
            float view = W12Codec.qViewElement(enc, 1, 64, 0, k);
            if (k == 3) {
                // the infinity's Q entry carries its scaled true value over the midpoint placeholder
                assertEquals(Float.floatToRawIntBits(Float.POSITIVE_INFINITY * scale),
                        Float.floatToRawIntBits(view));
                continue;
            }
            if (k == 0 || k == 2) {
                // the zero and the below-window outlier keep the midpoint placeholder, scaled with the
                // row: the view never recovers them, an error far inside the 8-bit view's half-binade bound
                assertEquals("placeholder at " + k, Float.floatToRawIntBits(W12Codec.Q_PLACEHOLDER * scale),
                        Float.floatToRawIntBits(view));
                continue;
            }
            int q = enc.get(k) & 0xFF;
            int s = (q >> 3) & 1;
            int c = (q >> 4) & 0xF;
            int m654 = q & 7;
            int bits = (s << 31) | ((c + top - 15) << 23) | (m654 << 20) | (W12Codec.QVIEW_NIBBLE << 16);
            assertEquals("element " + k, bits, Float.floatToRawIntBits(view));
            // the midpoint of the binade code differs from the exact element, whose B nibble is 0 here
            assertNotEquals(Float.floatToRawIntBits(exact[k]), Float.floatToRawIntBits(view));
        }
    }

    @Test
    public void theQViewCorrectsTheAboveWindowEscapesFromItsTable() {
        // the tie-to-highest top of the exponent-100 fillers is 115, so the exponent-140 weight escapes
        // above the window; its scaled value w * 2^(127 - 115) reaches the Q table and the view carries
        // it over the midpoint placeholder
        float[] values = new float[64];
        for (int k = 0; k < 64; k++) {
            values[k] = bf16((100 << 7) | ((k & 7) << 4));
        }
        values[5] = bf16(0x4640);
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
        int top = windowTop(enc, 64, 0);
        assertEquals(115, top);
        assertEquals(1, word(enc, W12Codec.planeBytes(64) + 8));
        float scale = Float.intBitsToFloat(top << 23);
        float expected = (values[5] * (float) Math.pow(2.0, 127 - 115)
                + 1.0625f * (float) Math.pow(2.0, -15)) * scale;
        assertEquals(expected, W12Codec.qViewElement(enc, 1, 64, 0, 5), 0.0f);
    }

    @Test
    public void qViewErrorStaysWithinHalfABinadeOfTheExactElement() {
        Random r = new Random(7);
        for (int trial = 0; trial < 16; trial++) {
            float[] values = new float[64];
            for (int k = 0; k < 64; k++) {
                int e = 120 + r.nextInt(10);
                values[k] = bf16((e << 7) | (r.nextInt(8) << 4) | r.nextInt(16));
            }
            ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, values, 1, 64));
            for (int k = 0; k < 64; k++) {
                float exact = values[k];
                float view = W12Codec.qViewElement(enc, 1, 64, 0, k);
                double ratio = Math.abs(view / exact);
                assertTrue("trial " + trial + " element " + k + ": view " + view + " for " + exact,
                        ratio >= 1.0 / 64.0 && ratio <= 64.0);
            }
        }
    }

    @Test
    public void qViewReadsAnyRowOfARecord() {
        float[] both = new float[128];
        for (int k = 0; k < 64; k++) {
            both[k] = 1.0f;
            both[64 + k] = bf16((200 << 7) | ((k & 7) << 4));
        }
        ByteBuffer enc = ByteBuffer.wrap(W12Codec.encode(T, both, 2, 64));
        int top1 = windowTop(enc, 64, 1);
        assertEquals(215, top1);
        // row 1, element 5: the view folds the row's scale, so the bits carry the element's own exponent
        // 200 with the midpoint nibble over its m654
        int bits = ((200 << 7) | (5 << 4) | W12Codec.QVIEW_NIBBLE) << 16;
        assertEquals(bits, Float.floatToRawIntBits(W12Codec.qViewElement(enc, 2, 64, 1, 5)));
        // row 0: 1.0 in band under the tie-to-highest top 142 (c = 15, m654 = 0), viewed at its own
        // exponent 127
        int bits0 = ((127 << 7) | W12Codec.QVIEW_NIBBLE) << 16;
        assertEquals(bits0, Float.floatToRawIntBits(W12Codec.qViewElement(enc, 2, 64, 0, 5)));
    }
}
