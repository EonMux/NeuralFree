package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK2_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK4_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_I8B128;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_M;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_M;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_QS;

/**
 * Bit-level primitives of the integer quantization formats: {@code b}-bit values packed in 32-bit words, the AWQ
 * nibble order, the stable act-order argsort, the fp16 representability test, element decoding of small safetensors
 * buffers, and encoders of the ggml block types the integer schemes target (bit-exact inverses of the
 * {@code dequantize_row_*} functions of llama.cpp ggml-quants.c). These are pure functions over caller arrays and
 * buffers; buffers passed in are little-endian and accessed by absolute index only.
 *
 * <p>On top of them, {@link CodeTable} and {@link ParamTable} read the codes and per-group parameters of a range of
 * output rows from each checkpoint layout (rows packed along K, GPTQ columns, AWQ nibbles, zero points packed along
 * N, row- and group-major parameters).</p>
 *
 * <p>Two word layouts exist ({@link Layout}): {@code 32/b} values per word, value {@code i} of a word in bits
 * {@code [i*b, i*b + b)} (GPTQ, compressed-tensors 0.15 and llama.cpp {@code dequant_packed}); and a continuous
 * stream in which element {@code e} starts at bit {@code e*b} of the little-endian word sequence and may straddle two
 * words (GPTQ 3-bit, vLLM's {@code ceil(K*b/32)} compressed-tensors sizing, MLX). Both coincide when {@code b}
 * divides 32. The packed word count selects the layout ({@link #layoutOf}).</p>
 */
public final class IntPacking {

    /** How {@code b}-bit values are laid out in a sequence of 32-bit words. */
    public enum Layout {
        /** {@code 32/b} values per word, least significant first; leftover high bits unused. */
        PER_WORD,
        /** Element {@code e} at bit {@code e*b} of the little-endian word stream. */
        BIT_STREAM
    }

    /** AutoAWQ order: slot {@code i} of a word holds column {@code AWQ_ORDER[i]} of its group of 8 (auto_awq.py:72-76). */
    private static final int[] AWQ_ORDER = {0, 2, 4, 6, 1, 3, 5, 7};
    /** The slot holding column {@code c} of a group of 8: the inverse of {@link #AWQ_ORDER}. */
    private static final int[] AWQ_SLOT = {0, 4, 1, 5, 2, 6, 3, 7};

    private IntPacking() {
    }

    // ------------------------------------------------------------------ words

    /** The low {@code bits} bits set, for {@code 1 <= bits <= 31}. */
    public static int mask(int bits) {
        return (1 << bits) - 1;
    }

    /** Values per word in the {@link Layout#PER_WORD} layout. */
    public static int perWord(int bits) {
        return 32 / bits;
    }

    /** Value {@code i} of {@code word}: {@code (word >>> (i*b)) & ((1 << b) - 1)}. */
    public static int slot(int word, int i, int bits) {
        return (word >>> (i * bits)) & mask(bits);
    }

    /** Words holding {@code count} values of {@code bits} bits in {@code layout}. */
    public static long wordsFor(long count, int bits, Layout layout) {
        if (layout == Layout.PER_WORD) {
            int per = perWord(bits);
            return (count + per - 1) / per;
        }
        return (Math.multiplyExact(count, (long) bits) + 31) / 32;
    }

    /**
     * The layout whose word count for {@code count} values is {@code words}, or null when neither layout has that
     * count. When both do, they store the values identically unless {@link #ambiguous} says otherwise, and
     * {@link Layout#PER_WORD} is returned.
     */
    public static Layout layoutOf(long count, long words, int bits) {
        if (wordsFor(count, bits, Layout.PER_WORD) == words) {
            return Layout.PER_WORD;
        }
        return wordsFor(count, bits, Layout.BIT_STREAM) == words ? Layout.BIT_STREAM : null;
    }

    /**
     * True when {@code words} words fit {@code count} values in both layouts while the layouts place some value
     * differently: {@code b} does not divide 32 and the values span more than one word.
     */
    public static boolean ambiguous(long count, long words, int bits) {
        return 32 % bits != 0 && count > perWord(bits)
                && wordsFor(count, bits, Layout.PER_WORD) == words
                && wordsFor(count, bits, Layout.BIT_STREAM) == words;
    }

    /** The index of the word holding the first bit of element {@code e}. */
    public static long firstWord(long e, int bits, Layout layout) {
        return layout == Layout.PER_WORD ? e / perWord(bits) : (e * bits) >>> 5;
    }

    /** The index of the word holding the last bit of element {@code e}. */
    public static long lastWord(long e, int bits, Layout layout) {
        return layout == Layout.PER_WORD ? e / perWord(bits) : (e * bits + bits - 1) >>> 5;
    }

    /**
     * Element {@code e} of a packed stream whose word {@code w} is {@code words[base + w * stride]}.
     *
     * @param bits 1 to 8
     */
    public static int element(int[] words, int base, int stride, Layout layout, int bits, long e) {
        return element(words, base, stride, layout, bits, e, 0);
    }

    /**
     * Element {@code e} of a packed stream of which only words {@code firstWord, firstWord + 1, ...} are present,
     * word {@code w} at {@code words[base + (w - firstWord) * stride]}.
     *
     * @param bits 1 to 8
     */
    public static int element(int[] words, int base, int stride, Layout layout, int bits, long e, long firstWord) {
        if (layout == Layout.PER_WORD) {
            int per = perWord(bits);
            return slot(words[base + (int) (e / per - firstWord) * stride], (int) (e % per), bits);
        }
        long bit = e * bits;
        int w = (int) ((bit >>> 5) - firstWord);
        int shift = (int) (bit & 31);
        int v = words[base + w * stride] >>> shift;
        if (shift + bits > 32) {
            v |= words[base + (w + 1) * stride] << (32 - shift);
        }
        return v & mask(bits);
    }

    /** The column of a group of 8 that AWQ slot {@code slot} holds. */
    public static int awqColumn(int slot) {
        return AWQ_ORDER[slot];
    }

    /** The AWQ slot that holds column {@code column} of a group of 8. */
    public static int awqSlot(int column) {
        return AWQ_SLOT[column];
    }

    // ------------------------------------------------------------------ permutations and representability

    /**
     * The stable argsort of {@code keys} (indices ordered by key, ties in index order), as a counting sort.
     *
     * @param keys values in {@code [0, keyCount)}
     * @throws IllegalArgumentException when a key is outside that range
     */
    public static int[] stableArgsort(int[] keys, int keyCount) {
        int[] start = new int[keyCount + 1];
        for (int i = 0; i < keys.length; i++) {
            int k = keys[i];
            if (k < 0 || k >= keyCount) {
                throw new IllegalArgumentException("key " + k + " at index " + i + " is outside [0, " + keyCount + ")");
            }
            start[k + 1]++;
        }
        for (int k = 0; k < keyCount; k++) {
            start[k + 1] += start[k];
        }
        int[] order = new int[keys.length];
        for (int i = 0; i < keys.length; i++) {
            order[start[keys[i]]++] = i;
        }
        return order;
    }

    /** True when {@code x} is stored unchanged in an IEEE half: {@code FpBlockDecoders.roundToFp16(x) == x}. */
    public static boolean fp16Exact(float x) {
        return Float.floatToRawIntBits(FpBlockDecoders.roundToFp16(x)) == Float.floatToRawIntBits(x);
    }

    // ------------------------------------------------------------------ small tensor elements

    /** True for the dtypes {@link #floatAt} decodes exactly: F16, BF16, F32, F64. */
    public static boolean isFloat(StDType dtype) {
        return dtype == StDType.F16 || dtype == StDType.BF16 || dtype == StDType.F32 || dtype == StDType.F64;
    }

    /** True for the integer dtypes {@link #longAt} decodes. */
    public static boolean isInteger(StDType dtype) {
        switch (dtype) {
            case I8: case U8: case I16: case U16: case I32: case U32: case I64:
                return true;
            default:
                return false;
        }
    }

    /**
     * Element {@code index} of a buffer of {@code dtype} values as a float: F16, BF16 and F32 exactly, F64 rounded to
     * nearest.
     *
     * @throws IllegalArgumentException for a dtype that is not {@link #isFloat}
     */
    public static float floatAt(ByteBuffer le, int index, StDType dtype) {
        switch (dtype) {
            case F16: return FpBlockDecoders.fp16ToFp32(le.getShort(index * 2));
            case BF16: return FpBlockDecoders.bf16ToFp32(le.getShort(index * 2));
            case F32: return le.getFloat(index * 4);
            case F64: return (float) le.getDouble(index * 8);
            default: throw new IllegalArgumentException("dtype " + dtype.headerName() + " is not a float dtype");
        }
    }

    /**
     * Element {@code index} of a buffer of integer {@code dtype} values.
     *
     * @throws IllegalArgumentException for a dtype that is not {@link #isInteger}
     */
    public static long longAt(ByteBuffer le, int index, StDType dtype) {
        switch (dtype) {
            case I8: return le.get(index);
            case U8: return le.get(index) & 0xFF;
            case I16: return le.getShort(index * 2);
            case U16: return le.getShort(index * 2) & 0xFFFF;
            case I32: return le.getInt(index * 4);
            case U32: return le.getInt(index * 4) & 0xFFFFFFFFL;
            case I64: return le.getLong(index * 8);
            default: throw new IllegalArgumentException("dtype " + dtype.headerName() + " is not an integer dtype");
        }
    }

    // ------------------------------------------------------------------ ggml block encoders

    /**
     * Writes one Q4_0 block: {@code d} as fp16 and the signed values {@code v[vOff .. vOff+32)}, each in
     * {@code [-8, 7]}, as nibbles {@code v + 8} (element j low, j+16 high; ggml-quants.c dequantize_row_q4_0).
     */
    public static void putQ4_0(ByteBuffer le, int off, float d, int[] v, int vOff) {
        le.putShort(off + Q4_0_D, (short) FpBlockDecoders.fp32ToFp16(d));
        int half = QK4_0 / 2;
        for (int j = 0; j < half; j++) {
            le.put(off + Q4_0_QS + j, (byte) ((v[vOff + j] + 8) | ((v[vOff + j + half] + 8) << 4)));
        }
    }

    /**
     * Writes one Q4_1 block: {@code d} and {@code m} as fp16 and the codes {@code q[qOff .. qOff+32)}, each in
     * {@code [0, 15]} (element value {@code q*d + m}; dequantize_row_q4_1).
     */
    public static void putQ4_1(ByteBuffer le, int off, float d, float m, int[] q, int qOff) {
        le.putShort(off + Q4_1_D, (short) FpBlockDecoders.fp32ToFp16(d));
        le.putShort(off + Q4_1_M, (short) FpBlockDecoders.fp32ToFp16(m));
        int half = QK4_0 / 2;
        for (int j = 0; j < half; j++) {
            le.put(off + Q4_1_QS + j, (byte) (q[qOff + j] | (q[qOff + j + half] << 4)));
        }
    }

    /**
     * Writes one Q5_0 block from signed values in {@code [-16, 15]} (codes {@code v + 16}; bit 4 of element j in bit
     * j of {@code qh}; dequantize_row_q5_0).
     */
    public static void putQ5_0(ByteBuffer le, int off, float d, int[] v, int vOff) {
        le.putShort(off + Q5_0_D, (short) FpBlockDecoders.fp32ToFp16(d));
        putQ5Bits(le, off + Q5_0_QH, off + Q5_0_QS, v, vOff, 16);
    }

    /** Writes one Q5_1 block from codes in {@code [0, 31]} (value {@code q*d + m}; dequantize_row_q5_1). */
    public static void putQ5_1(ByteBuffer le, int off, float d, float m, int[] q, int qOff) {
        le.putShort(off + Q5_1_D, (short) FpBlockDecoders.fp32ToFp16(d));
        le.putShort(off + Q5_1_M, (short) FpBlockDecoders.fp32ToFp16(m));
        putQ5Bits(le, off + Q5_1_QH, off + Q5_1_QS, q, qOff, 0);
    }

    private static void putQ5Bits(ByteBuffer le, int qhOff, int qsOff, int[] v, int vOff, int bias) {
        int half = QK4_0 / 2;
        int qh = 0;
        for (int j = 0; j < half; j++) {
            int lo = v[vOff + j] + bias;
            int hi = v[vOff + j + half] + bias;
            le.put(qsOff + j, (byte) ((lo & 0xF) | ((hi & 0xF) << 4)));
            qh |= ((lo >>> 4) & 1) << j;
            qh |= ((hi >>> 4) & 1) << (j + half);
        }
        le.putInt(qhOff, qh);
    }

    /** Writes one Q8_0 block from signed values in {@code [-128, 127]} (dequantize_row_q8_0). */
    public static void putQ8_0(ByteBuffer le, int off, float d, int[] v, int vOff) {
        le.putShort(off + Q8_0_D, (short) FpBlockDecoders.fp32ToFp16(d));
        for (int j = 0; j < QK8_0; j++) {
            le.put(off + Q8_0_QS + j, (byte) v[vOff + j]);
        }
    }

    /**
     * Writes one Q2_0 block from codes in {@code [0, 3]} (element value {@code (q - 1) * d}; element j in bits
     * {@code (j%4)*2} of byte {@code j/4}; ggml-quants.c:439-457).
     */
    public static void putQ2_0(ByteBuffer le, int off, float d, int[] q, int qOff) {
        le.putShort(off + Q2_0_D, (short) FpBlockDecoders.fp32ToFp16(d));
        for (int b = 0; b < QK2_0 / 4; b++) {
            int i = qOff + b * 4;
            le.put(off + Q2_0_QS + b, (byte) (q[i] | (q[i + 1] << 2) | (q[i + 2] << 4) | (q[i + 3] << 6)));
        }
    }

    /**
     * Writes one TQ2_0 block from codes in {@code [0, 3]} (element value {@code (q - 1) * d}): element
     * {@code e} sits in bits {@code 2*((e%128)/32)} of byte {@code 32*(e/128) + e%32}, the order in which
     * dequantize_row_tq2_0 (ggml-quants.c:2468-2485) emits elements.
     */
    public static void putTq2_0(ByteBuffer le, int off, float d, int[] q, int qOff) {
        for (int j = 0; j < QK_K / 4; j += 32) {
            for (int m = 0; m < 32; m++) {
                int e = qOff + j * 4 + m;
                le.put(off + TQ2_0_QS + j + m, (byte) (q[e] | (q[e + 32] << 2) | (q[e + 64] << 4) | (q[e + 96] << 6)));
            }
        }
        le.putShort(off + TQ2_0_D, (short) FpBlockDecoders.fp32ToFp16(d));
    }

    /** Writes one I8_B128 block: {@code d} as f32 and 128 signed values in {@code [-128, 127]}. */
    public static void putI8B128(ByteBuffer le, int off, float d, int[] v, int vOff) {
        le.putFloat(off + I8B128_D, d);
        for (int j = 0; j < QK_I8B128; j++) {
            le.put(off + I8B128_QS + j, (byte) v[vOff + j]);
        }
    }

    /**
     * Writes {@code x} as element {@code index} of a row of {@code type} at {@code off}: F32 as is, F16 and BF16
     * rounded to nearest even.
     *
     * @throws IllegalArgumentException for another type
     */
    public static void putFloat(ByteBuffer le, int off, int index, GGMLType type, float x) {
        switch (type) {
            case F32: le.putFloat(off + index * 4, x); break;
            case F16: le.putShort(off + index * 2, (short) FpBlockDecoders.fp32ToFp16(x)); break;
            case BF16: le.putShort(off + index * 2, (short) HalfFloats.floatToBFloat16(x)); break;
            default: throw new IllegalArgumentException("ggml type " + type + " is not a float type");
        }
    }

    // ------------------------------------------------------------------ packed checkpoint tensors

    /** Bytes of input rows read per batch by the readers that walk a whole tensor. */
    private static final int ROW_BATCH_BYTES = 4 << 20;

    /**
     * The integer codes of a quantized weight, read for ranges of output rows. Immutable and thread-safe; each read
     * allocates its own scratch.
     */
    static final class CodeTable {
        /** Reads the codes of rows {@code [n0, n0 + count)} of {@code expert}. */
        @FunctionalInterface
        interface Reader {
            void read(int expert, long n0, int count, byte[] out) throws IOException;
        }

        private final boolean rowLocal;
        private final Reader reader;

        CodeTable(boolean rowLocal, Reader reader) {
            this.rowLocal = rowLocal;
            this.reader = reader;
        }

        /**
         * Writes code {@code k} of row {@code n0 + i} to {@code out[i * K + k]} as an unsigned byte: a {@code b}-bit
         * code is {@code [0, 2^b)} with its zero point at {@code 2^(b-1)} when the format stores signed values.
         */
        void read(int expert, long n0, int count, byte[] out) throws IOException {
            reader.read(expert, n0, count, out);
        }

        /** True when reading a range of rows reads a proportional range of the stored tensor. */
        boolean rowLocal() {
            return rowLocal;
        }
    }

    /** Per-(row, group) parameters (scales, zero points, biases), read for ranges of output rows. Immutable. */
    static final class ParamTable {
        /** Reads the parameters of rows {@code [n0, n0 + count)} of {@code expert}. */
        @FunctionalInterface
        interface Reader {
            void read(int expert, long n0, int count, float[] out) throws IOException;
        }

        private final boolean rowLocal;
        private final Reader reader;

        ParamTable(boolean rowLocal, Reader reader) {
            this.rowLocal = rowLocal;
            this.reader = reader;
        }

        /** Writes the parameter of row {@code n0 + i} and group {@code g} to {@code out[i * groups + g]}. */
        void read(int expert, long n0, int count, float[] out) throws IOException {
            reader.read(expert, n0, count, out);
        }

        boolean rowLocal() {
            return rowLocal;
        }
    }

    /** Reads rows {@code [first, first + count)} of {@code src} into a new little-endian buffer. */
    static ByteBuffer rows(RowSource src, long first, int count) throws IOException {
        long bytes = Math.multiplyExact(src.rowBytes(), (long) count);
        if (bytes > Integer.MAX_VALUE - 8) {
            throw new IOException(src.describe() + ": " + count + " rows need " + bytes
                    + " bytes, more than one buffer holds");
        }
        ByteBuffer b = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(first, count, b);
        return b;
    }

    /** The innermost dimension of {@code src}; 1 for a scalar. */
    static long inner(RowSource src) {
        long[] s = src.shape();
        return s.length == 0 ? 1 : s[s.length - 1];
    }

    /**
     * Reads flat elements {@code [start, start + len)} of a tensor of {@code dtype} into {@code out}: float dtypes by
     * {@link #floatAt}, integer dtypes by {@link #longAt} converted to float.
     */
    static void readFloats(RowSource src, StDType dtype, long start, int len, float[] out, int outOff)
            throws IOException {
        if (len == 0) {
            return;
        }
        long in = inner(src);
        long r0 = start / in;
        long r1 = (start + len - 1) / in;
        ByteBuffer b = rows(src, r0, Math.toIntExact(r1 - r0 + 1));
        int skip = (int) (start - r0 * in);
        for (int i = 0; i < len; i++) {
            out[outOff + i] = isFloat(dtype) ? floatAt(b, skip + i, dtype) : (float) longAt(b, skip + i, dtype);
        }
    }

    /** Reads flat 32-bit words {@code [start, start + len)} of an I32 or U32 tensor into {@code out}. */
    static void readWords(RowSource src, long start, int len, int[] out, int outOff) throws IOException {
        if (len == 0) {
            return;
        }
        long in = inner(src);
        long r0 = start / in;
        long r1 = (start + len - 1) / in;
        ByteBuffer b = rows(src, r0, Math.toIntExact(r1 - r0 + 1));
        int skip = (int) (start - r0 * in);
        for (int i = 0; i < len; i++) {
            out[outOff + i] = b.getInt((skip + i) * 4);
        }
    }

    /** Parameters stored row-major, {@code [E?, N, G]} or any shape with {@code N*G} elements per expert. */
    static ParamTable rowMajor(RowSource src, long n, int groups) {
        StDType dt = src.format().stDType();
        return new ParamTable(true, (e, n0, count, out) ->
                readFloats(src, dt, (e * n + n0) * groups, count * groups, out, 0));
    }

    /** One value per expert, or one in all, broadcast over rows and groups. */
    static ParamTable broadcast(RowSource src, long numel, int groups) {
        StDType dt = src.format().stDType();
        return new ParamTable(true, (e, n0, count, out) -> {
            float[] v = new float[1];
            readFloats(src, dt, numel == 1 ? 0 : e, 1, v, 0);
            Arrays.fill(out, 0, count * groups, v[0]);
        });
    }

    /** {@code t} with {@code add} added to every value (signed zero points expressed against offset codes). */
    static ParamTable shifted(ParamTable t, float add) {
        return new ParamTable(t.rowLocal(), (e, n0, count, out) -> {
            t.read(e, n0, count, out);
            for (int i = 0; i < out.length; i++) {
                out[i] += add;
            }
        });
    }

    /** Parameters stored group-major, {@code [E?, G, N]} (GPTQ and AWQ {@code scales}). */
    static ParamTable groupMajor(RowSource src, long n, int groups) {
        StDType dt = src.format().stDType();
        return new ParamTable(false, (e, n0, count, out) -> {
            float[] row = new float[count];
            for (int g = 0; g < groups; g++) {
                readFloats(src, dt, ((long) e * groups + g) * n + n0, count, row, 0);
                for (int i = 0; i < count; i++) {
                    out[i * groups + g] = row[i];
                }
            }
        });
    }

    /**
     * Rows {@code [E?, R, cols]} of 32-bit words packed along the row (compressed-tensors {@code weight_packed}, MLX
     * {@code weight}): row {@code n} holds the {@code k} values of output row {@code n}.
     */
    static CodeTable rowPacked(RowSource src, long rowsPerExpert, int k, int bits, Layout layout) {
        int cols = Math.toIntExact(inner(src));
        return new CodeTable(true, (e, n0, count, out) -> {
            ByteBuffer b = rows(src, e * rowsPerExpert + n0, count);
            int[] words = new int[cols];
            for (int i = 0; i < count; i++) {
                for (int c = 0; c < cols; c++) {
                    words[c] = b.getInt((i * cols + c) * 4);
                }
                for (int kk = 0; kk < k; kk++) {
                    out[i * k + kk] = (byte) element(words, 0, 1, layout, bits, kk);
                }
            }
        });
    }

    /**
     * GPTQ {@code qweight} {@code [E?, Kp, N]}: the words of output row {@code n} run down column {@code n}. Every
     * read walks all {@code Kp} rows.
     */
    static CodeTable columnPacked(RowSource src, int kp, long n, int k, int bits, Layout layout) {
        int batch = (int) Math.max(1, Math.min(kp, ROW_BATCH_BYTES / (n * 4)));
        return new CodeTable(false, (e, n0, count, out) -> {
            int[] cw = new int[Math.multiplyExact(count, kp)];
            for (int r0 = 0; r0 < kp; r0 += batch) {
                int rc = Math.min(batch, kp - r0);
                ByteBuffer b = rows(src, (long) e * kp + r0, rc);
                for (int r = 0; r < rc; r++) {
                    long rowBase = r * n + n0;
                    for (int i = 0; i < count; i++) {
                        cw[i * kp + r0 + r] = b.getInt((int) ((rowBase + i) * 4));
                    }
                }
            }
            for (int i = 0; i < count; i++) {
                for (int kk = 0; kk < k; kk++) {
                    out[i * k + kk] = (byte) element(cw, i * kp, 1, layout, bits, kk);
                }
            }
        });
    }

    /**
     * AWQ {@code qweight} or Quark int4 {@code weight} {@code [E?, K, Np]}: the nibble of output row {@code n} in row
     * {@code k} sits in word {@code n/8}, slot {@code awqSlot(n%8)} (AWQ order) or {@code n%8}, and is XOR-ed with
     * {@code xor}. Every read walks all {@code K} rows.
     */
    static CodeTable nibblesAlongN(RowSource src, int k, int np, boolean awqOrder, int xor) {
        int batch = Math.max(1, Math.min(k, ROW_BATCH_BYTES / (np * 4)));
        return new CodeTable(false, (e, n0, count, out) -> {
            for (int r0 = 0; r0 < k; r0 += batch) {
                int rc = Math.min(batch, k - r0);
                ByteBuffer b = rows(src, (long) e * k + r0, rc);
                for (int r = 0; r < rc; r++) {
                    for (int i = 0; i < count; i++) {
                        long nn = n0 + i;
                        int w = b.getInt((int) (((long) r * np + nn / 8) * 4));
                        int c = (int) (nn & 7);
                        int s = awqOrder ? awqSlot(c) : c;
                        out[i * k + r0 + r] = (byte) (((w >>> (s * 4)) & 0xF) ^ xor);
                    }
                }
            }
        });
    }

    /**
     * GPTQ {@code qzeros} {@code [E?, G, Nw]}: row {@code g} holds the zero points of every output row packed along
     * N; {@code add} is added to each (1 for GPTQ v1).
     */
    static ParamTable packedRows(RowSource src, int groups, int words, int bits, Layout layout, int add) {
        return new ParamTable(false, (e, n0, count, out) -> {
            long w0 = firstWord(n0, bits, layout);
            int nw = (int) (lastWord(n0 + count - 1, bits, layout) - w0 + 1);
            int[] buf = new int[nw];
            for (int g = 0; g < groups; g++) {
                readWords(src, ((long) e * groups + g) * words + w0, nw, buf, 0);
                for (int i = 0; i < count; i++) {
                    out[i * groups + g] = element(buf, 0, 1, layout, bits, n0 + i, w0) + add;
                }
            }
        });
    }

    /**
     * compressed-tensors {@code weight_zero_point} {@code [E?, Nw, G]}: the zero points of group {@code g} run down
     * column {@code g}, packed along N.
     */
    static ParamTable packedColumns(RowSource src, long wordRows, int groups, int bits, Layout layout) {
        return new ParamTable(true, (e, n0, count, out) -> {
            long w0 = firstWord(n0, bits, layout);
            int nw = (int) (lastWord(n0 + count - 1, bits, layout) - w0 + 1);
            int[] buf = new int[Math.multiplyExact(nw, groups)];
            readWords(src, ((long) e * wordRows + w0) * groups, nw * groups, buf, 0);
            for (int i = 0; i < count; i++) {
                for (int g = 0; g < groups; g++) {
                    out[i * groups + g] = element(buf, g, groups, layout, bits, n0 + i, w0);
                }
            }
        });
    }

    /**
     * AWQ {@code qzeros} or Quark {@code weight_zero_point} {@code [E?, G, Np]}: nibbles along N, each XOR-ed with
     * {@code xor}.
     */
    static ParamTable nibbleRows(RowSource src, int groups, int np, boolean awqOrder, int xor) {
        return new ParamTable(false, (e, n0, count, out) -> {
            long w0 = n0 / 8;
            int nw = (int) ((n0 + count - 1) / 8 - w0 + 1);
            int[] buf = new int[nw];
            for (int g = 0; g < groups; g++) {
                readWords(src, ((long) e * groups + g) * np + w0, nw, buf, 0);
                for (int i = 0; i < count; i++) {
                    long nn = n0 + i;
                    int c = (int) (nn & 7);
                    int s = awqOrder ? awqSlot(c) : c;
                    out[i * groups + g] = ((buf[(int) (nn / 8 - w0)] >>> (s * 4)) & 0xF) ^ xor;
                }
            }
        });
    }
}
