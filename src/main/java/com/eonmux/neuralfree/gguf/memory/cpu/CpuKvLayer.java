package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.util.Arrays;
import java.util.Objects;

/**
 * Heap KV cache of one owning layer. K and V rows are stored in their {@link KvDType}: F32 as {@code float[]}, F16 and
 * BF16 as {@code short[]} (IEEE half / bfloat16 bits, round to nearest even), the block types (Q8_0, Q4_0, Q4_1,
 * IQ4_NL, Q5_0, Q5_1) as {@code byte[]} ggml blocks quantized by the reference quantizers of {@link KvBlockCodec}.
 * Caches whose element count exceeds a Java array are split into row chunks. Reads may run concurrently with each
 * other; writes and row moves are made by the thread that owns the step.
 */
public final class CpuKvLayer implements KvAccessor {

    private final int layer;
    private final KvLayout layout;
    private final int group;
    private final int nHeadKv;
    private final int headDimK;
    private final int headDimV;
    private final Rows k;
    private final Rows v;

    private CpuKvLayer(int layer, KvLayout layout, int group, int nHeadKv, int headDimK, int headDimV) {
        this.layer = layer;
        this.layout = layout;
        this.group = group;
        this.nHeadKv = nHeadKv;
        this.headDimK = headDimK;
        this.headDimV = headDimV;
        this.k = new Rows(layout.dtypeK(), layout.rows(), layout.widthK());
        this.v = layout.vAliasesK() ? null : new Rows(layout.dtypeV(), layout.rows(), layout.widthV());
    }

    /**
     * Allocates a zeroed cache. A latent layout ({@link KvLayout#vAliasesK()}) has one head of
     * {@code (widthK, widthV)}; otherwise the widths must equal {@code nHeadKv * headDimK} and
     * {@code nHeadKv * headDimV}.
     *
     * @param group KV group addressing the rows, or -1 for a cache not addressed by a group
     * @throws IllegalArgumentException when the head shape does not match the layout widths
     */
    public static CpuKvLayer allocate(int layer, KvLayout layout, int group, int nHeadKv, int headDimK, int headDimV) {
        Objects.requireNonNull(layout, "layout");
        if (layout.vAliasesK()) {
            if (nHeadKv != 1 || headDimK != layout.widthK() || headDimV != layout.widthV()) {
                throw new IllegalArgumentException("latent KV cache of layer " + layer + " has one head of ("
                        + layout.widthK() + ", " + layout.widthV() + "), got " + nHeadKv + " x (" + headDimK + ", "
                        + headDimV + ")");
            }
        } else if (nHeadKv <= 0 || (long) nHeadKv * headDimK != layout.widthK()
                || (long) nHeadKv * headDimV != layout.widthV()) {
            throw new IllegalArgumentException("KV cache of layer " + layer + " stores rows of K " + layout.widthK()
                    + " / V " + layout.widthV() + ", which " + nHeadKv + " heads of (" + headDimK + ", " + headDimV
                    + ") do not fill");
        }
        return new CpuKvLayer(layer, layout, group, nHeadKv, headDimK, headDimV);
    }

    /** A single-head key-only cache (indexer, MSA and compressed rows): V reads the K row and cannot be written. */
    static CpuKvLayer keyOnly(int layer, KvLayout latentLayout, int group) {
        return allocate(layer, latentLayout, group, 1, latentLayout.widthK(), latentLayout.widthV());
    }

    @Override
    public int layer() {
        return layer;
    }

    @Override
    public KvLayout layout() {
        return layout;
    }

    @Override
    public int group() {
        return group;
    }

    @Override
    public int nHeadKv() {
        return nHeadKv;
    }

    @Override
    public int headDimK() {
        return headDimK;
    }

    @Override
    public int headDimV() {
        return headDimV;
    }

    @Override
    public void readK(int row, int head, float[] dst, int off) {
        checkHead(head);
        k.read(row, head * headDimK, headDimK, dst, off);
    }

    @Override
    public void readV(int row, int head, float[] dst, int off) {
        checkHead(head);
        if (v == null) {
            k.read(row, 0, headDimV, dst, off);
        } else {
            v.read(row, head * headDimV, headDimV, dst, off);
        }
    }

    @Override
    public void writeK(int row, float[] src, int off) {
        k.write(row, src, off);
    }

    @Override
    public void writeKSlice(int row, int from, int n, float[] src, int off) {
        k.writeSlice(row, from, n, src, off);
    }

    @Override
    public void writeV(int row, float[] src, int off) {
        if (v == null) {
            throw new UnsupportedOperationException("V of the latent cache of layer " + layer + " is part of the K row");
        }
        v.write(row, src, off);
    }

    /** Applies a context shift's row moves to K and (when separate) V. */
    void moveRows(DeviceStepInputs.KvShift shift) {
        k.moveRows(shift);
        if (v != null) {
            v.moveRows(shift);
        }
    }

    /** Zeroes every row. */
    void clear() {
        k.clear();
        if (v != null) {
            v.clear();
        }
    }

    private void checkHead(int head) {
        if (head < 0 || head >= nHeadKv) {
            throw new IndexOutOfBoundsException("head " + head + " outside [0, " + nHeadKv + ") of layer " + layer);
        }
    }

    @Override
    public String toString() {
        return "CpuKvLayer{il=" + layer + ", group=" + group + ", heads=" + nHeadKv + "x(" + headDimK + ", " + headDimV
                + "), " + layout + '}';
    }

    // ================================================================== row storage

    /** Rows of one dtype, chunked so no array exceeds the JVM limit. */
    static final class Rows {

        private static final int MAX_ARRAY = Integer.MAX_VALUE - 8;

        private final KvDType dtype;
        private final int rows;
        private final int width;
        private final int unitsPerRow;
        private final int rowsPerChunk;
        private final float[][] f32;
        private final short[][] h16;
        private final byte[][] blocks;

        Rows(KvDType dtype, int rows, int width) {
            this.dtype = Objects.requireNonNull(dtype, "dtype");
            if (rows <= 0 || width <= 0 || !dtype.supportsWidth(width)) {
                throw new IllegalArgumentException(dtype + " rows of width " + width + " x " + rows + " are invalid");
            }
            this.rows = rows;
            this.width = width;
            this.unitsPerRow = dtype.quantized() ? width / dtype.blockSize() * dtype.blockBytes() : width;
            this.rowsPerChunk = Math.max(1, MAX_ARRAY / unitsPerRow);
            int chunks = (int) (((long) rows + rowsPerChunk - 1) / rowsPerChunk);
            this.f32 = dtype == KvDType.F32 ? new float[chunks][] : null;
            this.h16 = dtype == KvDType.F16 || dtype == KvDType.BF16 ? new short[chunks][] : null;
            this.blocks = dtype.quantized() ? new byte[chunks][] : null;
            for (int c = 0; c < chunks; c++) {
                int n = Math.min(rowsPerChunk, rows - c * rowsPerChunk) * unitsPerRow;
                if (f32 != null) {
                    f32[c] = new float[n];
                } else if (h16 != null) {
                    h16[c] = new short[n];
                } else {
                    blocks[c] = new byte[n];
                }
            }
        }

        int rows() {
            return rows;
        }

        int width() {
            return width;
        }

        /** Dequantizes values {@code [from, from + n)} of {@code row} into {@code dst[off..]}. */
        void read(int row, int from, int n, float[] dst, int off) {
            checkRow(row);
            checkSlice(from, n);
            if (off < 0 || n > dst.length - off) {
                throw new IndexOutOfBoundsException("destination [" + off + ", " + (off + n) + ") outside "
                        + dst.length);
            }
            int c = row / rowsPerChunk;
            int base = (row - c * rowsPerChunk) * unitsPerRow;
            switch (dtype) {
                case F32:
                    System.arraycopy(f32[c], base + from, dst, off, n);
                    break;
                case F16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        dst[off + i] = FpBlockDecoders.fp16ToFp32(a[base + from + i]);
                    }
                    break;
                }
                case BF16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        dst[off + i] = FpBlockDecoders.bf16ToFp32(a[base + from + i]);
                    }
                    break;
                }
                default: {
                    final int bs = dtype.blockSize();
                    int i = 0;
                    while (i < n) {
                        int e = from + i;
                        int j = e % bs;
                        int run = Math.min(bs - j, n - i);
                        KvBlockCodec.dequantize(dtype, blocks[c], base + (e / bs) * dtype.blockBytes(), j, run, dst,
                                off + i);
                        i += run;
                    }
                    break;
                }
            }
        }

        /** Stores a full row of {@code width} values from {@code src[off..]}. */
        void write(int row, float[] src, int off) {
            writeSlice(row, 0, width, src, off);
        }

        /**
         * Stores values {@code [from, from + n)} of {@code row} from {@code src[off..]}; the other values keep their
         * stored bits. A block type needs {@code from} and {@code n} on block boundaries.
         *
         * @throws IllegalArgumentException for a block-type slice that splits a block
         */
        void writeSlice(int row, int from, int n, float[] src, int off) {
            checkRow(row);
            checkSlice(from, n);
            if (off < 0 || n > src.length - off) {
                throw new IndexOutOfBoundsException("source [" + off + ", " + (off + n) + ") outside " + src.length);
            }
            int c = row / rowsPerChunk;
            int base = (row - c * rowsPerChunk) * unitsPerRow;
            switch (dtype) {
                case F32:
                    System.arraycopy(src, off, f32[c], base + from, n);
                    break;
                case F16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        a[base + from + i] = (short) FpBlockDecoders.fp32ToFp16(src[off + i]);
                    }
                    break;
                }
                case BF16: {
                    short[] a = h16[c];
                    for (int i = 0; i < n; i++) {
                        a[base + from + i] = (short) KvBlockCodec.fp32ToBf16(src[off + i]);
                    }
                    break;
                }
                default: {
                    final int bs = dtype.blockSize();
                    if (from % bs != 0 || n % bs != 0) {
                        throw new IllegalArgumentException(dtype + " rows are written in whole blocks of " + bs
                                + " values, got values [" + from + ", " + (from + n) + ")");
                    }
                    for (int b = 0; b < n / bs; b++) {
                        KvBlockCodec.quantize(dtype, src, off + b * bs, blocks[c],
                                base + (from / bs + b) * dtype.blockBytes());
                    }
                    break;
                }
            }
        }

        private void checkSlice(int from, int n) {
            if (from < 0 || n < 0 || from > width - n) {
                throw new IndexOutOfBoundsException("values [" + from + ", " + (from + n) + ") outside a row of "
                        + width);
            }
        }

        /** Zeroes every row. */
        void clear() {
            int chunks = f32 != null ? f32.length : h16 != null ? h16.length : blocks.length;
            for (int c = 0; c < chunks; c++) {
                if (f32 != null) {
                    Arrays.fill(f32[c], 0f);
                } else if (h16 != null) {
                    Arrays.fill(h16[c], (short) 0);
                } else {
                    Arrays.fill(blocks[c], (byte) 0);
                }
            }
        }

        /**
         * Executes a context shift's row moves: downward moves row by row in ascending order (each row is read
         * before any move writes it); otherwise the moved rows are gathered first and then scattered.
         */
        void moveRows(DeviceStepInputs.KvShift shift) {
            if (shift.runCount() == 0) {
                return;
            }
            if (!shift.needsStaging()) {
                for (int i = 0; i < shift.runCount(); i++) {
                    for (int r = 0; r < shift.runRows(i); r++) {
                        copyRow(shift.runSource(i) + r, shift.runTarget(i) + r);
                    }
                }
                return;
            }
            Object[] staged = new Object[shift.movedRows()];
            int at = 0;
            for (int i = 0; i < shift.runCount(); i++) {
                for (int r = 0; r < shift.runRows(i); r++) {
                    staged[at++] = rowCopy(shift.runSource(i) + r);
                }
            }
            at = 0;
            for (int i = 0; i < shift.runCount(); i++) {
                for (int r = 0; r < shift.runRows(i); r++) {
                    putRow(shift.runTarget(i) + r, staged[at++]);
                }
            }
        }

        private void copyRow(int src, int dst) {
            putRow(dst, chunk(src), offset(src));
        }

        private Object rowCopy(int row) {
            Object a = chunk(row);
            int o = offset(row);
            if (f32 != null) {
                return Arrays.copyOfRange((float[]) a, o, o + unitsPerRow);
            }
            if (h16 != null) {
                return Arrays.copyOfRange((short[]) a, o, o + unitsPerRow);
            }
            return Arrays.copyOfRange((byte[]) a, o, o + unitsPerRow);
        }

        private void putRow(int row, Object data) {
            putRow(row, data, 0);
        }

        private void putRow(int row, Object src, int srcOff) {
            System.arraycopy(src, srcOff, chunk(row), offset(row), unitsPerRow);
        }

        private Object chunk(int row) {
            checkRow(row);
            int c = row / rowsPerChunk;
            return f32 != null ? f32[c] : h16 != null ? h16[c] : blocks[c];
        }

        private int offset(int row) {
            return (row % rowsPerChunk) * unitsPerRow;
        }

        private void checkRow(int row) {
            if (row < 0 || row >= rows) {
                throw new IndexOutOfBoundsException("row " + row + " outside a cache of " + rows + " rows");
            }
        }
    }
}
