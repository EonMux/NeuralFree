package com.eonmux.neuralfree.gguf.hf.torchao;

import com.eonmux.neuralfree.gguf.hf.quant.IntPacking;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.utils.numeric.Fp8Codec;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Produced row sources for decoded torchao tensors: the plain two-nibbles-per-byte weight of a tile-packed
 * {@code Int4TilePackedTo4dTensor}, an mslk preshuffled {@code Int4PreshuffledTensor} or an XPU
 * {@code Int4PlainInt32Tensor}, the scale and zero point separated from a {@code scale_and_zero} pack, the group
 * scale recovered from the FP8-activation preshuffle's eight-value lookup, the affine zero point of an unpacked
 * {@code IntxUnpackedToInt8Tensor}, the unswizzled scale grid of a swizzled MX/NVFP4 tensor, and constant values.
 * They implement the ordinary {@link RowSource} contract so the quantization schemes read them like any checkpoint
 * tensor.
 */
final class TorchaoSources {

    private TorchaoSources() {
    }

    /** A source over {@code values}, F32, in the given shape. */
    static RowSource floats(float[] values, long[] shape, String describe) {
        return new Floats(values, shape, describe);
    }

    /** A source of zeros in the given shape, F32. */
    static RowSource zeros(long[] shape, String describe) {
        long count = numel(shape);
        return new Floats(new float[Math.toIntExact(count)], shape, describe);
    }

    /**
     * {@code raw}, whose rows and columns are in the NVIDIA blocked (swizzled) scale layout, as an ordinary grid of
     * the same dtype: the leading dimensions {@code lead} (the matrix rows, or experts and rows), then one column per
     * 4-column scale block.
     */
    static RowSource unblocked(RowSource raw, long[] lead, long groups, String describe) {
        return new Unblocked(raw, lead, groups, describe);
    }

    /**
     * A tile-packed int4 weight ({@code [ceil(n/8), ceil(k/(I*16)), 32, I/2]} I32, or the same with a leading expert
     * dimension, as {@code _convert_weight_to_int4pack} writes it for {@code inner_k_tiles = I}) as plain
     * {@code [n, k/2]} U8 with two signed int4 values per byte, element {@code 2j} in the low nibble. The CUDA packing
     * writes {@code inner_k_tiles} 8, and the ROCm backend pads N to 16 without changing the layout; I 2, 4, 8 and 16
     * decode through the same formula.
     */
    static RowSource tilePacked(RowSource qdata, long n, long k, long experts, String describe) {
        return new TilePacked(qdata, n, k, experts, describe);
    }

    /**
     * A plain-int32 int4 weight ({@code [n, k/8]} I32, four packed bytes per word as XPU's
     * {@code _convert_weight_to_int4pack} writes it) as plain {@code [n, k/2]} U8: little-endian byte order already
     * holds elements {@code 2j} and {@code 2j+1} in one byte.
     */
    static RowSource plainInt32(RowSource qdata, long n, long k, long[] shape, String describe) {
        return new PlainInt32(qdata, n, k, shape, describe);
    }

    /**
     * A preshuffled int4 weight (mslk {@code preshuffle_i4}) as plain {@code [n, k/2]} U8: the inverse of the
     * CUTLASS {@code compute_memory_reordering_atom} permutation, and of the {@code unified_encode_int4b} code map
     * when {@code fp8}. The bf16-activation layout reorders 16x16 atoms; the fp8-activation one reorders 16x32 atoms
     * and additionally encodes each code for the kernel's table lookup.
     */
    static RowSource preshuffled(RowSource qdata, long n, long k, long experts, boolean fp8, String describe) {
        return new Preshuffled(qdata, n, k, experts, fp8, describe);
    }

    /**
     * A {@code [n, k]} I8 tensor of sub-byte codes stored one per byte, packed two per U8 byte with element
     * {@code 2j} in the low nibble.
     */
    static RowSource packNibbles(RowSource codes, long n, long k, long experts, String describe) {
        return new PackNibbles(codes, n, k, experts, describe);
    }

    /**
     * The {@code scale} or {@code zero} half of a packed {@code [groups, rows, 2]} (or the same with a leading expert
     * dimension) tinygemm {@code scale_and_zero} tensor, as the same float dtype and the logical {@code [E?, G, N]}
     * shape. torchao pads the tile-packed weight's rows to 8 and its K to 1024, so the stored scale grid is larger
     * than the weight's.
     */
    static RowSource scaleOrZero(RowSource packed, boolean zero, long groups, long n, long experts,
                                 String describe) {
        return new ScaleOrZero(packed, zero, groups, n, experts, describe);
    }

    /** Constant F32 values. */
    private static final class Floats implements RowSource {
        private final float[] values;
        private final long[] shape;
        private final String describe;
        private final long inner;
        private final long rows;

        Floats(float[] values, long[] shape, String describe) {
            this.values = values.clone();
            this.shape = shape.clone();
            this.describe = describe;
            long numel = 1;
            for (long d : shape) {
                numel = Math.multiplyExact(numel, d);
            }
            if (numel != values.length) {
                throw new IllegalArgumentException(describe + " has " + values.length + " values for shape "
                        + Arrays.toString(shape));
            }
            this.inner = shape.length == 0 ? 1 : shape[shape.length - 1];
            this.rows = inner == 0 ? 0 : numel / inner;
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.F32); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return rows; }
        @Override public long rowBytes() { return Math.multiplyExact(inner, 4L); }
        @Override public boolean rowLocal() { return true; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            long from = Math.multiplyExact(first, inner);
            long n = Math.multiplyExact((long) count, inner);
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside " + describe);
            }
            for (long i = 0; i < n; i++) {
                dst.putFloat(values[(int) (from + i)]);
            }
        }
    }

    /** One blocked scale grid read as its logical row-major grid. */
    private static final class Unblocked implements RowSource {
        private static final long MAX_CACHED = 1L << 29;
        private final RowSource raw;
        private final long[] lead;
        private final long n;
        private final long groups;
        private final long rowsPerExpert;
        private final long experts;
        private final long gridElements;
        private final String describe;
        private final StDType dtype;
        private final int elemBytes;
        private final long colBlocks;
        private byte[] cached;

        Unblocked(RowSource raw, long[] lead, long groups, String describe) {
            this.raw = Objects.requireNonNull(raw, "raw");
            this.lead = lead.clone();
            if (lead.length == 0) {
                throw new IllegalArgumentException(describe + " has no row dimension");
            }
            this.n = numel(lead);
            this.groups = groups;
            this.rowsPerExpert = lead[lead.length - 1];
            this.experts = lead.length == 1 ? 1 : numel(Arrays.copyOf(lead, lead.length - 1));
            this.describe = describe;
            this.dtype = raw.format().stDType();
            if (dtype == null || dtype.bits() % 8 != 0) {
                throw new IllegalArgumentException(describe + " has no byte-addressable scale elements: "
                        + raw.format());
            }
            this.elemBytes = dtype.bits() / 8;
            this.colBlocks = (groups + 3) / 4;
            long rowBlocks = (rowsPerExpert + 127) / 128;
            this.gridElements = Math.multiplyExact(rowBlocks * colBlocks, 512L);
            long expectedElements = Math.multiplyExact(experts, gridElements);
            long actual = numel(raw.shape());
            if (actual < expectedElements) {
                throw new IllegalArgumentException(describe + ": the blocked scale tensor has " + actual
                        + " elements; " + n + " rows of " + groups + " scales need " + expectedElements);
            }
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(dtype); }
        @Override public long[] shape() {
            long[] s = Arrays.copyOf(lead, lead.length + 1);
            s[lead.length] = groups;
            return s;
        }
        @Override public long rows() { return n; }
        @Override public long rowBytes() { return Math.multiplyExact(groups, (long) elemBytes); }
        @Override public boolean rowLocal() { return false; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            if (first < 0 || count < 0 || first > n - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside " + describe);
            }
            byte[] b = bytes();
            for (long r = first; r < first + count; r++) {
                long e = r / rowsPerExpert;
                long rIn = r % rowsPerExpert;
                for (long c = 0; c < groups; c++) {
                    int src = (int) ((e * gridElements + blockedIndex(rIn, c)) * elemBytes);
                    for (int k = 0; k < elemBytes; k++) {
                        dst.put(b[src + k]);
                    }
                }
            }
        }

        private long blockedIndex(long r, long c) {
            long rowBlock = r / 128;
            long x = (r % 128) / 32;
            long y = r % 32;
            long colBlock = c / 4;
            long z = c % 4;
            return (rowBlock * colBlocks + colBlock) * 512 + y * 16 + x * 4 + z;
        }

        private byte[] bytes() throws IOException {
            synchronized (this) {
                if (cached == null) {
                    long size = numel(raw.shape());
                    long limit = MAX_CACHED / elemBytes;
                    if (size > limit) {
                        throw new IOException(describe + " has " + size + " elements, more than the "
                                + limit + " a swizzled scale tensor may hold");
                    }
                    ByteBuffer b = ByteBuffer.allocate((int) Math.multiplyExact(size, (long) elemBytes))
                            .order(ByteOrder.LITTLE_ENDIAN);
                    raw.readRows(0, Math.toIntExact(raw.rows()), b);
                    cached = b.array();
                }
                return cached;
            }
        }
    }

    /** A plain two-nibble weight decoded from the tensor-core tile packing. */
    private static final class TilePacked implements RowSource {
        private final RowSource qdata;
        private final long n;
        private final long k;
        private final long experts;
        private final String describe;
        private final long[] shape;
        private final int innerTiles;
        private final int kBlocks;
        private final int rowBlocks;
        private final int bytesPerRow;
        private final int kBlockBytes;
        private final int wordsPerRow;
        private byte[] slab;
        private long slabBlock = -1;

        TilePacked(RowSource qdata, long n, long k, long experts, String describe) {
            this.qdata = Objects.requireNonNull(qdata, "qdata");
            this.n = n;
            this.k = k;
            this.experts = experts;
            this.describe = describe;
            long[] qshape = qdata.shape();
            int lead = experts > 0 ? 1 : 0;
            if (qshape.length != lead + 4 || qshape[lead + 2] != 32 || qshape[lead + 3] < 1
                    || qshape[lead + 3] > 8) {
                throw new IllegalArgumentException(describe + " has the tile-packed shape "
                        + Arrays.toString(qshape) + "; expected [n/8, k/(inner_k_tiles*16), 32,"
                        + " inner_k_tiles/2] with inner_k_tiles 2, 4, 8 or 16"
                        + (experts > 0 ? ", behind a leading expert dimension" : ""));
            }
            if (experts > 0 && qshape[0] != experts) {
                throw new IllegalArgumentException(describe + " packs " + qshape[0] + " experts, not " + experts);
            }
            if (k % 2 != 0) {
                throw new IllegalArgumentException(describe + " has an odd K = " + k);
            }
            this.innerTiles = Math.toIntExact(qshape[lead + 3] * 2);
            this.kBlocks = Math.toIntExact(qshape[lead + 1]);
            this.rowBlocks = Math.toIntExact(qshape[lead]);
            this.bytesPerRow = Math.toIntExact(k / 2);
            this.kBlockBytes = innerTiles * 8;
            this.wordsPerRow = innerTiles / 2;
            if (Math.multiplyExact((long) kBlocks, (long) innerTiles * 16) < k) {
                throw new IllegalArgumentException(describe + ": " + kBlocks + " K blocks hold "
                        + (kBlocks * (long) innerTiles * 16) + " elements, fewer than K = " + k);
            }
            if (Math.multiplyExact((long) rowBlocks, 8L) < n) {
                throw new IllegalArgumentException(describe + ": " + rowBlocks + " row blocks hold "
                        + (rowBlocks * 8L) + " rows, fewer than N = " + n);
            }
            this.shape = experts > 0 ? new long[]{experts, n, k / 2} : new long[]{n, k / 2};
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.U8); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return Math.max(1, experts) * n; }
        @Override public long rowBytes() { return bytesPerRow; }
        @Override public boolean rowLocal() { return true; }

        @Override
        public synchronized void readRows(long first, int count, ByteBuffer dst) throws IOException {
            long rows = rows();
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside " + describe);
            }
            int start = dst.position();
            for (long r = first; r < first + count; r++) {
                long block = r / 8;
                byte[] b = slab(block);
                int rIn = (int) (r % 8);
                int base = (int) (r - first) * bytesPerRow;
                for (int c = 0; c < k / 2; c++) {
                    int kBlock = c / kBlockBytes;
                    int local = c % kBlockBytes;
                    int tile = local / 16;
                    int c4 = c % 4;
                    int j = (local % 16) / 4;
                    int at = (kBlock * 32 + 4 * rIn + c4) * (wordsPerRow * 4) + tile * 4;
                    int word = (b[at] & 0xFF) | ((b[at + 1] & 0xFF) << 8) | ((b[at + 2] & 0xFF) << 16)
                            | ((b[at + 3] & 0xFF) << 24);
                    int even = ((word >>> (4 * j)) & 0xF) ^ 0x8;
                    int odd = ((word >>> (4 * (4 + j))) & 0xF) ^ 0x8;
                    dst.put(start + base + c, (byte) (even | (odd << 4)));
                }
            }
            dst.position(start + count * bytesPerRow);
        }

        /** The 32-row slab of packed word {@code block}, which is one (expert, 8-row, K-block) group. */
        private byte[] slab(long block) throws IOException {
            if (slabBlock != block) {
                long perExpert = (long) rowBlocks * kBlocks;
                long e = block / perExpert;
                long local = block % perExpert;
                long firstRow = ((e * rowBlocks + local) * kBlocks) * 32;
                int rows = Math.multiplyExact(kBlocks, 32);
                int rowBytes = Math.multiplyExact(wordsPerRow, 4);
                ByteBuffer b = ByteBuffer.allocate(Math.multiplyExact(rows, rowBytes))
                        .order(ByteOrder.LITTLE_ENDIAN);
                qdata.readRows(firstRow, rows, b);
                slab = b.array();
                slabBlock = block;
            }
            return slab;
        }
    }

    /** One half (scale or zero) of a packed scale_and_zero tensor. */
    private static final class ScaleOrZero implements RowSource {
        private final RowSource packed;
        private final boolean zero;
        private final String describe;
        private final long[] shape;
        private final StDType dtype;
        private final int elemBytes;
        private final long columns;
        private final long storedGroups;
        private final long groups;
        private final long n;
        private final long experts;
        private final long rows;

        ScaleOrZero(RowSource packed, boolean zero, long groups, long n, long experts, String describe) {
            this.packed = Objects.requireNonNull(packed, "packed");
            this.zero = zero;
            this.describe = describe;
            this.experts = experts;
            long[] p = packed.shape();
            int lead = experts > 0 ? 1 : 0;
            if (p.length != lead + 3 || p[lead + 2] != 2) {
                throw new IllegalArgumentException(describe + " has shape " + Arrays.toString(p)
                        + "; expected a scale_and_zero tensor of [groups, rows, 2]"
                        + (experts > 0 ? " behind a leading expert dimension" : ""));
            }
            if (experts > 0 && p[0] != experts) {
                throw new IllegalArgumentException(describe + " packs " + p[0] + " experts, not " + experts);
            }
            if (groups <= 0 || n <= 0 || p[lead] < groups || p[lead + 1] < n) {
                throw new IllegalArgumentException(describe + " has shape " + Arrays.toString(p) + "; it does not"
                        + " hold " + groups + " groups of " + n + " rows");
            }
            this.shape = countShape(experts, groups, n);
            this.dtype = packed.format().stDType();
            if (dtype == null || dtype.bits() % 8 != 0) {
                throw new IllegalArgumentException(describe + " has no byte-addressable elements: "
                        + packed.format());
            }
            this.elemBytes = dtype.bits() / 8;
            this.columns = p[lead + 1];
            this.storedGroups = p[lead];
            this.groups = groups;
            this.n = n;
            this.rows = Math.max(1, experts) * groups;
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(dtype); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return rows; }
        @Override public long rowBytes() { return Math.multiplyExact(n, (long) elemBytes); }
        @Override public boolean rowLocal() { return true; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside " + describe);
            }
            int pairBytes = Math.multiplyExact(2, elemBytes);
            int pick = zero ? 1 : 0;
            int capacityPairs = Math.max(1, (1 << 20) / pairBytes);
            ByteBuffer b = ByteBuffer.allocate(Math.multiplyExact(capacityPairs, pairBytes))
                    .order(ByteOrder.LITTLE_ENDIAN);
            for (long row = first; row < first + count; row++) {
                long pairs = n;
                long e = experts > 0 ? row / groups : 0;
                long g = experts > 0 ? row % groups : row;
                long start = Math.multiplyExact(Math.addExact(Math.multiplyExact(e, storedGroups), g), columns);
                for (long done = 0; done < pairs; done += capacityPairs) {
                    int c = (int) Math.min(capacityPairs, pairs - done);
                    b.clear();
                    ByteBuffer window = b.duplicate();
                    window.limit(Math.multiplyExact(c, pairBytes));
                    packed.readRows(start + done, c, window);
                    byte[] a = b.array();
                    for (int i = 0; i < c; i++) {
                        int at = i * pairBytes + pick * elemBytes;
                        for (int k = 0; k < elemBytes; k++) {
                            dst.put(a[at + k]);
                        }
                    }
                }
            }
        }
    }

    /** A weight carrying a per-column activation pre-scale, to fold into the scheme's float fallback. */
    static RowSource preScaled(RowSource weight, float[] preScale, String describe) {
        return new PreScaled(weight, preScale, describe);
    }

    /**
     * The zero-point tensor of an {@code IntxUnpackedToInt8Tensor} weight in the layout
     * {@link com.eonmux.neuralfree.gguf.hf.config.QuantFormat#INT4_TORCHAO} reads: {@code -z * s} per group and row,
     * from the row-major {@code [n, groups]} scale and zero point. The stored code is signed {@code q}, which the
     * scheme offsets to {@code q + 8}; with the bias {@code z' - 8s} the scheme computes
     * {@code (q + 8)s + (-z*s - 8s) = (q - z)s}, torchao's affine dequantization.
     */
    static RowSource intxBias(RowSource scale, RowSource zeroPoint, long n, int groups, long experts,
                              String describe) {
        return new Bias(scale, zeroPoint, n, groups, experts, false, false, describe);
    }

    /**
     * The zero-point tensor of an XPU {@code Int4PlainInt32Tensor}: {@code (8 - z) * s} in group-major order. The
     * decoded weight holds the two's complement of the unsigned code {@code u}, which the scheme offsets back to
     * {@code u}; the scheme's bias {@code z' - 8s} then leaves {@code u*s - z*s = (u - z)s}.
     */
    static RowSource plainInt32Bias(RowSource scale, RowSource zeroPoint, long n, int groups, long experts,
                                    String describe) {
        return new Bias(scale, zeroPoint, n, groups, experts, true, true, describe);
    }

    /** A row-major {@code [n, groups]} float scale as the group-major {@code [groups, n]} grid. */
    static RowSource groupMajorFloats(RowSource scale, long n, int groups, long experts, String describe) {
        return new Transposed(scale, n, groups, experts, describe);
    }

    /**
     * The logical group scale of an FP8-activation preshuffled weight: {@code row_scale[n] * s} where {@code s} is
     * the normalized E4M3 scale stored in the eight-value lookup, whose last entry is {@code -s}.
     */
    static RowSource preshuffleFp8Scale(RowSource packed, RowSource rowScale, long n, int groups, long experts,
                                        String describe) {
        return new Fp8LookupScale(packed, rowScale, n, groups, experts, describe);
    }

    /** The values of a small float tensor as Java floats. */
    static float[] floats(RowSource src, String describe) throws IOException {
        StDType dtype = src.format().stDType();
        if (dtype == null || !IntPacking.isFloat(dtype)) {
            throw new IllegalArgumentException(describe + " has no float dtype: " + src.format());
        }
        ByteBuffer b = all(src);
        long count = Math.multiplyExact(src.rows(), src.rowBytes()) / (dtype.bits() / 8);
        float[] out = new float[Math.toIntExact(count)];
        for (int i = 0; i < out.length; i++) {
            out[i] = IntPacking.floatAt(b, i, dtype);
        }
        return out;
    }

    /** The whole tensor as a little-endian buffer, refusing inputs too large to hold. */
    private static ByteBuffer all(RowSource s) throws IOException {
        long rows = s.rows();
        long rowBytes = s.rowBytes();
        long bytes = Math.multiplyExact(rows, rowBytes);
        if (bytes > MAX_CACHED_BYTES) {
            throw new IOException(s.describe() + " has " + bytes + " bytes, more than the " + MAX_CACHED_BYTES
                    + " a decoded scale tensor may hold");
        }
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(bytes)).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, Math.toIntExact(rows), b);
        b.flip();
        return b;
    }

    private static float floatValue(ByteBuffer b, long index, StDType dtype, String describe) {
        int i = Math.toIntExact(index);
        if (dtype == StDType.F8_E4M3) {
            return Fp8Codec.E4M3FN.decode(b.get(i) & 0xFF);
        }
        return IntPacking.floatAt(b, i, dtype);
    }

    private static float numberValue(ByteBuffer b, long index, StDType dtype, String describe) {
        int i = Math.toIntExact(index);
        if (IntPacking.isFloat(dtype)) {
            return IntPacking.floatAt(b, i, dtype);
        }
        if (IntPacking.isInteger(dtype)) {
            return IntPacking.longAt(b, i, dtype);
        }
        throw new IllegalArgumentException(describe + " has no numeric dtype: " + dtype.headerName());
    }

    private static long numel(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    private static long[] countShape(long experts, long groups, long n) {
        return experts > 0 ? new long[]{experts, groups, n} : new long[]{groups, n};
    }

    /** The largest tensor {@link #all} materializes. */
    private static final long MAX_CACHED_BYTES = 1L << 31;

    /** A plain int4 weight stored as int32 words of four packed bytes. */
    private static final class PlainInt32 implements RowSource {
        private final RowSource qdata;
        private final long[] shape;
        private final long rows;
        private final long rowBytes;
        private final String describe;

        PlainInt32(RowSource qdata, long n, long k, long[] shape, String describe) {
            this.qdata = Objects.requireNonNull(qdata, "qdata");
            this.describe = describe;
            if (qdata.format().stDType() != StDType.I32) {
                throw new IllegalArgumentException(describe + " is stored as " + qdata.format()
                        + "; expected I32 words of four packed int4 bytes");
            }
            if (k % 8 != 0) {
                throw new IllegalArgumentException(describe + " has K = " + k + ", not a whole number of"
                        + " 8-value int32 words");
            }
            long[] q = qdata.shape();
            if (q.length != shape.length || q[q.length - 1] != k / 8 || q[q.length - 2] != shape[shape.length - 2]) {
                throw new IllegalArgumentException(describe + " has qdata of shape " + Arrays.toString(q)
                        + "; expected " + Arrays.toString(shape) + " with the last dimension " + (k / 8)
                        + " int32 words");
            }
            this.shape = shape.clone();
            this.rows = qdata.rows();
            this.rowBytes = k / 2;
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.U8); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return rows; }
        @Override public long rowBytes() { return rowBytes; }
        @Override public boolean rowLocal() { return qdata.rowLocal(); }
        @Override public long directVirtualOffset() { return qdata.directVirtualOffset(); }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            int start = dst.position();
            qdata.readRows(first, count, dst);
            // The XPU codes are unsigned 0..15; the plain two-nibble layout holds their two's complement, so the
            // shared engine's code table offsets them back. XOR with 8 maps u to u - 8 across the whole byte.
            for (int i = start; i < dst.position(); i++) {
                dst.put(i, (byte) (dst.get(i) ^ 0x88));
            }
        }
    }

    /** Sub-byte codes stored one per I8 byte, packed two per output byte. */
    private static final class PackNibbles implements RowSource {
        private final RowSource codes;
        private final long n;
        private final long k;
        private final long experts;
        private final long[] shape;
        private final String describe;

        PackNibbles(RowSource codes, long n, long k, long experts, String describe) {
            this.codes = Objects.requireNonNull(codes, "codes");
            this.n = n;
            this.k = k;
            this.experts = experts;
            this.describe = describe;
            if (codes.format().stDType() != StDType.I8) {
                throw new IllegalArgumentException(describe + " is stored as " + codes.format()
                        + "; expected I8 codes, one per element");
            }
            long[] q = codes.shape();
            int lead = experts > 0 ? 1 : 0;
            if (q.length != lead + 2 || q[lead] != n || q[lead + 1] != k) {
                throw new IllegalArgumentException(describe + " has codes of shape " + Arrays.toString(q)
                        + "; expected " + (experts > 0 ? "[" + experts + ", " : "[") + n + ", " + k + "]");
            }
            if (k % 2 != 0) {
                throw new IllegalArgumentException(describe + " has an odd K = " + k);
            }
            this.shape = experts > 0 ? new long[]{experts, n, k / 2} : new long[]{n, k / 2};
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.U8); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return Math.max(1, experts) * n; }
        @Override public long rowBytes() { return k / 2; }
        @Override public boolean rowLocal() { return true; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            long rows = rows();
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside " + describe);
            }
            int start = dst.position();
            int perRow = Math.toIntExact(k);
            ByteBuffer in = ByteBuffer.allocate(Math.multiplyExact(count, perRow)).order(ByteOrder.LITTLE_ENDIAN);
            codes.readRows(first, count, in);
            byte[] a = in.array();
            for (int r = 0; r < count; r++) {
                int b = start + r * (perRow / 2);
                for (int j = 0; j < perRow; j += 2) {
                    dst.put(b + j / 2, (byte) ((a[r * perRow + j] & 0xF) | ((a[r * perRow + j + 1] & 0xF) << 4)));
                }
            }
            dst.position(start + count * (perRow / 2));
        }
    }

    /** A preshuffled int4 weight read back through the CUTLASS reordering atom's inverse. */
    private static final class Preshuffled implements RowSource {
        private final RowSource qdata;
        private final long n;
        private final long k;
        private final long experts;
        private final boolean fp8;
        private final long[] shape;
        private final long nTiles;
        private final int kAtom;
        private final int tileSize;
        private final int bytesPerRow;
        private final String describe;
        private byte[] cached;

        Preshuffled(RowSource qdata, long n, long k, long experts, boolean fp8, String describe) {
            this.qdata = Objects.requireNonNull(qdata, "qdata");
            this.n = n;
            this.k = k;
            this.experts = experts;
            this.fp8 = fp8;
            this.describe = describe;
            // The bf16-activation atom spans 16 columns, the fp8-activation atom 32 (an E4M3 MMA reads a K of 32).
            this.kAtom = fp8 ? 32 : 16;
            this.tileSize = fp8 ? 512 : 256;
            if (n <= 0 || k <= 0 || n % 16 != 0 || k % kAtom != 0) {
                throw new IllegalArgumentException(describe + " has the shape [" + n + ", " + k + "]; the"
                        + (fp8 ? " FP8-activation" : " bf16-activation") + " preshuffle atom tiles 16 rows and "
                        + kAtom + " columns, so N must be a multiple of 16 and K of " + kAtom);
            }
            StDType t = qdata.format().stDType();
            if (t != StDType.I8 && t != StDType.U8) {
                throw new IllegalArgumentException(describe + " is stored as " + qdata.format()
                        + "; expected I8 or U8 packed int4 values");
            }
            long rows = Math.max(1, experts) * n;
            if (qdata.rows() != rows || qdata.rowBytes() != k / 2) {
                throw new IllegalArgumentException(describe + " has qdata of " + qdata.rows() + " rows of "
                        + qdata.rowBytes() + " bytes; expected " + rows + " rows of " + (k / 2));
            }
            this.shape = experts > 0 ? new long[]{experts, n, k / 2} : new long[]{n, k / 2};
            this.nTiles = n / 16;
            this.bytesPerRow = Math.toIntExact(k / 2);
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.U8); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return Math.max(1, experts) * n; }
        @Override public long rowBytes() { return bytesPerRow; }
        @Override public boolean rowLocal() { return false; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            long rows = rows();
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside " + describe);
            }
            byte[] a = bytes();
            int start = dst.position();
            long expertBytes = n * bytesPerRow;
            for (long r = first; r < first + count; r++) {
                long e = r / n;
                long row = r % n;
                int base = start + (int) (r - first) * bytesPerRow;
                long expertBase = e * expertBytes;
                for (int j = 0; j < bytesPerRow; j++) {
                    int q0 = decode(nibble(a, expertBase, index(row, 2L * j)));
                    int q1 = decode(nibble(a, expertBase, index(row, 2L * j + 1)));
                    dst.put(base + j, (byte) ((q0 & 0xF) | ((q1 & 0xF) << 4)));
                }
            }
            dst.position(start + count * bytesPerRow);
        }

        /**
         * The destination int4 position of logical element {@code (row, col)} in its expert's weight: the inverse of
         * {@code cutlass::compute_memory_reordering_atom} for the activation dtype's MMA operand, tiled over
         * 16-row atoms with the row tiles innermost.
         */
        private long index(long row, long col) {
            long atom;
            if (fp8) {
                atom = 64 * (row % 8) + 4 * ((row / 8) % 2)
                        + (col % 4) + 16 * ((col / 4) % 4) + 8 * ((col % 32) / 16);
            } else {
                atom = 32 * (row % 8) + ((row / 8) % 2)
                        + 4 * (col % 2) + 8 * ((col / 2) % 4) + 2 * ((col / 8) % 2);
            }
            return (long) tileSize * ((row / 16) + nTiles * (col / kAtom)) + atom;
        }

        private static int nibble(byte[] a, long expertBase, long position) {
            int value = a[Math.toIntExact(expertBase + position / 2)] & 0xFF;
            return (value >>> (4 * (int) (position & 1))) & 0xF;
        }

        /** The signed int4 of a stored nibble: {@code unified_encode_int4b}'s code map on the FP8 path only. */
        private int decode(int code) {
            if (!fp8) {
                return code >= 8 ? code - 16 : code;
            }
            if (code == 0) {
                return 0;
            }
            return code < 8 ? 8 - code : code - 16;
        }

        private synchronized byte[] bytes() throws IOException {
            if (cached == null) {
                ByteBuffer b = all(qdata);
                cached = b.array();
            }
            return cached;
        }
    }

    /** One weight with a per-column activation pre-scale attached. */
    private static final class PreScaled implements RowSource {
        private final RowSource weight;
        private final float[] preScale;
        private final String describe;

        PreScaled(RowSource weight, float[] preScale, String describe) {
            this.weight = Objects.requireNonNull(weight, "weight");
            this.preScale = Objects.requireNonNull(preScale, "preScale").clone();
            this.describe = describe;
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return weight.format(); }
        @Override public long[] shape() { return weight.shape(); }
        @Override public long rows() { return weight.rows(); }
        @Override public long rowBytes() { return weight.rowBytes(); }
        @Override public boolean rowLocal() { return weight.rowLocal(); }
        @Override public long directVirtualOffset() { return weight.directVirtualOffset(); }
        @Override public float[] inputPreScale() { return preScale.clone(); }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            weight.readRows(first, count, dst);
        }
    }

    /** {@code (8 - z) * s} or {@code -z * s} per group and row, in group-major or row-major source order. */
    private static final class Bias implements RowSource {
        private final RowSource scale;
        private final RowSource zeroPoint;
        private final long n;
        private final int groups;
        private final long experts;
        private final boolean groupMajor;
        private final boolean offsetCodes;
        private final long[] shape;
        private final String describe;
        private float[] values;

        Bias(RowSource scale, RowSource zeroPoint, long n, int groups, long experts, boolean groupMajor,
             boolean offsetCodes, String describe) {
            this.scale = Objects.requireNonNull(scale, "scale");
            this.zeroPoint = Objects.requireNonNull(zeroPoint, "zeroPoint");
            this.n = n;
            this.groups = groups;
            this.experts = experts;
            this.groupMajor = groupMajor;
            this.offsetCodes = offsetCodes;
            this.describe = describe;
            StDType sd = scale.format().stDType();
            StDType zd = zeroPoint.format().stDType();
            if (sd == null || !IntPacking.isFloat(sd)) {
                throw new IllegalArgumentException(describe + " has a scale of " + scale.format());
            }
            if (zd == null || (!IntPacking.isFloat(zd) && !IntPacking.isInteger(zd))) {
                throw new IllegalArgumentException(describe + " has a zero point of " + zeroPoint.format());
            }
            this.shape = countShape(experts, groups, n);
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.F32); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return Math.max(1, experts) * groups; }
        @Override public long rowBytes() { return Math.multiplyExact(n, 4L); }
        @Override public boolean rowLocal() { return true; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            float[] v = values();
            long inner = n;
            long from = first * inner;
            for (long i = 0; i < count * inner; i++) {
                dst.putFloat(v[Math.toIntExact(from + i)]);
            }
        }

        private synchronized float[] values() throws IOException {
            if (values != null) {
                return values;
            }
            StDType sd = scale.format().stDType();
            StDType zd = zeroPoint.format().stDType();
            long count = Math.multiplyExact(Math.multiplyExact(Math.max(1, experts), groups), n);
            ByteBuffer s = all(scale);
            ByteBuffer z = all(zeroPoint);
            if (s.capacity() < count * sd.bits() / 8 || z.capacity() < count * zd.bits() / 8) {
                throw new IllegalArgumentException(describe + ": the scale and zero point do not hold " + count
                        + " values");
            }
            float[] out = new float[Math.toIntExact(count)];
            int at = 0;
            for (long e = 0; e < Math.max(1, experts); e++) {
                for (int gi = 0; gi < groups; gi++) {
                    for (long r = 0; r < n; r++) {
                        long index = groupMajor
                                ? (e * groups + gi) * n + r
                                : (e * n + r) * groups + gi;
                        float sc = floatValue(s, index, sd, describe);
                        float zp = numberValue(z, index, zd, describe);
                        out[at++] = offsetCodes ? (8.0f - zp) * sc : -zp * sc;
                    }
                }
            }
            values = out;
            return values;
        }
    }

    /** A row-major {@code [n, groups]} float grid as the group-major {@code [groups, n]} grid, F32. */
    private static final class Transposed implements RowSource {
        private final RowSource scale;
        private final long n;
        private final int groups;
        private final long experts;
        private final long[] shape;
        private final String describe;
        private float[] values;

        Transposed(RowSource scale, long n, int groups, long experts, String describe) {
            this.scale = Objects.requireNonNull(scale, "scale");
            this.n = n;
            this.groups = groups;
            this.experts = experts;
            this.describe = describe;
            StDType sd = scale.format().stDType();
            if (sd == null || !IntPacking.isFloat(sd)) {
                throw new IllegalArgumentException(describe + " has a scale of " + scale.format());
            }
            this.shape = countShape(experts, groups, n);
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.F32); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return Math.max(1, experts) * groups; }
        @Override public long rowBytes() { return Math.multiplyExact(n, 4L); }
        @Override public boolean rowLocal() { return true; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            float[] v = values();
            long from = first * n;
            for (long i = 0; i < count * n; i++) {
                dst.putFloat(v[Math.toIntExact(from + i)]);
            }
        }

        private synchronized float[] values() throws IOException {
            if (values != null) {
                return values;
            }
            StDType sd = scale.format().stDType();
            long count = Math.multiplyExact(Math.multiplyExact(Math.max(1, experts), groups), n);
            ByteBuffer b = all(scale);
            if (b.capacity() < count * sd.bits() / 8) {
                throw new IllegalArgumentException(describe + ": the scale does not hold " + count + " values");
            }
            float[] out = new float[Math.toIntExact(count)];
            int at = 0;
            for (long e = 0; e < Math.max(1, experts); e++) {
                for (int gi = 0; gi < groups; gi++) {
                    for (long r = 0; r < n; r++) {
                        out[at++] = floatValue(b, (e * n + r) * groups + gi, sd, describe);
                    }
                }
            }
            values = out;
            return values;
        }
    }

    /** {@code row_scale[n] * (-lookup[7])} per group and row of a preshuffled FP8-activation weight. */
    private static final class Fp8LookupScale implements RowSource {
        private final RowSource packed;
        private final RowSource rowScale;
        private final long n;
        private final int groups;
        private final long experts;
        private final long[] shape;
        private final String describe;
        private float[] values;

        Fp8LookupScale(RowSource packed, RowSource rowScale, long n, int groups, long experts, String describe) {
            this.packed = Objects.requireNonNull(packed, "packed");
            this.rowScale = Objects.requireNonNull(rowScale, "rowScale");
            this.n = n;
            this.groups = groups;
            this.experts = experts;
            this.describe = describe;
            if (packed.format().stDType() != StDType.F8_E4M3) {
                throw new IllegalArgumentException(describe + " has a packed scale of " + packed.format()
                        + "; expected E4M3");
            }
            StDType rd = rowScale.format().stDType();
            if (rd == null || !IntPacking.isFloat(rd)) {
                throw new IllegalArgumentException(describe + " has a row scale of " + rowScale.format());
            }
            this.shape = countShape(experts, groups, n);
        }

        @Override public String describe() { return describe; }
        @Override public ElementFormat format() { return ElementFormat.of(StDType.F32); }
        @Override public long[] shape() { return shape.clone(); }
        @Override public long rows() { return Math.max(1, experts) * groups; }
        @Override public long rowBytes() { return Math.multiplyExact(n, 4L); }
        @Override public boolean rowLocal() { return true; }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            float[] v = values();
            long from = first * n;
            for (long i = 0; i < count * n; i++) {
                dst.putFloat(v[Math.toIntExact(from + i)]);
            }
        }

        private synchronized float[] values() throws IOException {
            if (values != null) {
                return values;
            }
            StDType rd = rowScale.format().stDType();
            long count = Math.multiplyExact(Math.multiplyExact(Math.max(1, experts), groups), n);
            ByteBuffer p = all(packed);
            ByteBuffer r = all(rowScale);
            if (p.capacity() < count * 8 || r.capacity() < Math.max(1, experts) * n * rd.bits() / 8) {
                throw new IllegalArgumentException(describe + ": the packed and row scales do not cover "
                        + count + " values");
            }
            float[] out = new float[Math.toIntExact(count)];
            int at = 0;
            for (long e = 0; e < Math.max(1, experts); e++) {
                for (int gi = 0; gi < groups; gi++) {
                    for (long row = 0; row < n; row++) {
                        // The packed lookup holds eight byte values per (group, row) in memory order
                        // [expert][group][row][value]; the last is -scale.
                        long index = ((e * groups + gi) * n + row) * 8 + 7;
                        float normalized = -floatValue(p, index, StDType.F8_E4M3, describe);
                        float rowScaleValue = floatValue(r, e * n + row, rd, describe);
                        out[at++] = normalized * rowScaleValue;
                    }
                }
            }
            values = out;
            return values;
        }
    }
}
