package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The Engram tables of the DeepSeek-V4.1 converter (deepseek41.py:215-250, :277-285): the per-layer embedding rows
 * dequantized from the checkpoint's E4M3 codes and E8M0 block scales, and the multiplication of an Engram query table
 * by its key table (the elementwise gate the converter folds from two BF16 tensors into one F32 tensor).
 *
 * <p>An Engram embedding row is {@code code * 2^(e - 127)} with one scale byte per 32 elements; the product is exact
 * in BF16 for {@code e in [3, 246]} (the converter's {@code _e8m0_check}), so the dequantized rows are written as
 * BF16 or, with {@code --outtype f32}, F32 exactly as the converter rounds them.</p>
 */
public final class EngramRows {

    /** One layer's stored Engram table: the E4M3 codes and the E8M0 block scales. */
    public static final class Part {
        private final RowSource weight;
        private final RowSource scale;
        private final long rows;
        private final int dim;

        /**
         * @param weight the E4M3 table, {@code [rows, dim]}
         * @param scale  the E8M0 scales, {@code [rows, dim / 32]}
         * @param rows   the number of rows the configuration declares
         * @param dim    the row width
         */
        public Part(RowSource weight, RowSource scale, long rows, int dim) {
            this.weight = Objects.requireNonNull(weight, "weight");
            this.scale = Objects.requireNonNull(scale, "scale");
            if (rows < 0 || dim <= 0 || dim % 32 != 0) {
                throw new IllegalArgumentException("Engram part must have a non-negative row count and a width"
                        + " that is a multiple of 32, got " + rows + " rows of " + dim);
            }
            this.rows = rows;
            this.dim = dim;
        }

        public RowSource weight() { return weight; }

        public RowSource scale() { return scale; }

        public long rows() { return rows; }

        public int dim() { return dim; }
    }

    private EngramRows() {
    }

    /**
     * The concatenation of the parts' dequantized rows in part order.
     *
     * @param parts  the layers' tables, in order
     * @param output {@link GGMLType#BF16} or {@link GGMLType#F32}
     * @throws IllegalArgumentException when no part is given, their widths differ or the output type is neither
     *                                  BF16 nor F32
     */
    public static RowSource table(List<Part> parts, GGMLType output) {
        Objects.requireNonNull(parts, "parts");
        if (parts.isEmpty()) {
            throw new IllegalArgumentException("an Engram table needs at least one layer");
        }
        if (output != GGMLType.BF16 && output != GGMLType.F32) {
            throw new IllegalArgumentException("an Engram table is BF16 or F32, not " + output);
        }
        return new Table(Collections.unmodifiableList(new ArrayList<>(parts)), output);
    }

    /**
     * The elementwise product {@code a * b} in F32 (the converter's {@code q_weight.float() * k_weight.float()}).
     *
     * @throws IllegalArgumentException when the shapes differ or either source is not a matrix
     */
    public static RowSource product(RowSource a, RowSource b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        long[] shape = a.shape();
        if (shape.length != 2 || !java.util.Arrays.equals(shape, b.shape())) {
            throw new IllegalArgumentException("an Engram gate needs two matrices of the same shape, got "
                    + java.util.Arrays.toString(shape) + " and " + java.util.Arrays.toString(b.shape()));
        }
        return new Product(a, b);
    }

    // ------------------------------------------------------------------ producers

    private static final class Table extends RowTransforms.Derived {
        private final List<Part> parts;
        private final GGMLType output;
        private final long[] starts;

        Table(List<Part> parts, GGMLType output) {
            super(tableDescribe(parts, output), ElementFormat.of(outputDtype(output)), tableShape(parts),
                    ExpertTransforms.Siblings.NONE);
            this.parts = parts;
            this.output = output;
            this.starts = new long[parts.size()];
            long at = 0;
            for (int i = 0; i < parts.size(); i++) {
                starts[i] = at;
                at += parts.get(i).rows();
            }
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            int part = partAt(first);
            long row = first;
            int done = 0;
            while (done < count) {
                Part p = parts.get(part);
                long offset = row - starts[part];
                if (offset >= p.rows()) {
                    part++;
                    continue;
                }
                int inPart = (int) Math.min(count - done, p.rows() - offset);
                int max = Math.max(1, RowTransforms.CHUNK_BYTES / (p.dim * units()));
                for (int at = 0; at < inPart; at += max) {
                    int rows = Math.min(max, inPart - at);
                    dequantise(p, offset + at, rows, out);
                }
                done += inPart;
                row += inPart;
                part++;
            }
        }

        private void dequantise(Part p, long row, int rows, ByteBuffer out) throws IOException {
            ElementFormat weightFormat = p.weight().format();
            ElementFormat scaleFormat = p.scale().format();
            requireDtype(p.weight(), weightFormat, StDType.F8_E4M3, "weight");
            requireDtype(p.scale(), scaleFormat, StDType.U8, "scale");
            long[] scaleShape = p.scale().shape();
            if (scaleShape.length != 2 || scaleShape[0] != p.rows() || scaleShape[1] != p.dim() / 32) {
                throw new GgufFormatException(p.scale().describe() + ": expected " + p.rows() + " rows of "
                        + (p.dim() / 32) + " E8M0 scales (one per 32 elements), got "
                        + java.util.Arrays.toString(scaleShape));
            }
            long[] weightShape = p.weight().shape();
            if (weightShape.length != 2 || weightShape[0] != p.rows() || weightShape[1] != p.dim()) {
                throw new GgufFormatException(p.weight().describe() + ": expected " + p.rows() + " rows of "
                        + p.dim() + " E4M3 values, got " + java.util.Arrays.toString(weightShape));
            }
            ByteBuffer weights = ByteBuffer.allocate(Math.multiplyExact(rows, p.dim()))
                    .order(ByteOrder.LITTLE_ENDIAN);
            p.weight().readRows(row, rows, weights);
            weights.flip();
            ByteBuffer scales = ByteBuffer.allocate(Math.multiplyExact(rows, p.dim() / 32))
                    .order(ByteOrder.LITTLE_ENDIAN);
            p.scale().readRows(row, rows, scales);
            scales.flip();
            for (int r = 0; r < rows; r++) {
                int exponent = 0;
                for (int c = 0; c < p.dim(); c++) {
                    if (c % 32 == 0) {
                        exponent = scales.get() & 0xFF;
                        if (exponent < 3 || exponent > 246) {
                            throw new GgufFormatException(p.scale().describe() + ": E8M0 exponent " + exponent
                                    + " is outside [3, 246]; the dequantized values would not be exact");
                        }
                    }
                    float value = FpBlockDecoders.e4m3ToFp32(weights.get() & 0xFF) * scale(exponent);
                    if (output == GGMLType.BF16) {
                        out.putShort((short) HalfFloats.floatToBFloat16(value));
                    } else {
                        out.putFloat(value);
                    }
                }
            }
        }

        private int partAt(long row) {
            int low = 0;
            int high = parts.size() - 1;
            while (low < high) {
                int mid = (low + high + 1) >>> 1;
                if (starts[mid] <= row) {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            return low;
        }

        private int units() {
            return output == GGMLType.BF16 ? 2 : 4;
        }
    }

    private static final class Product extends RowTransforms.Derived {
        private final RowSource a;
        private final RowSource b;

        Product(RowSource a, RowSource b) {
            super(a.describe() + " * " + b.describe(), ElementFormat.of(StDType.F32), a.shape(),
                    ExpertTransforms.Siblings.NONE);
            this.a = a;
            this.b = b;
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            int inner = (int) a.shape()[1];
            int max = Math.max(1, RowTransforms.CHUNK_BYTES / (inner * 4));
            for (int at = 0; at < count; at += max) {
                int rows = Math.min(max, count - at);
                float[] va = values(a, first + at, rows, inner);
                float[] vb = values(b, first + at, rows, inner);
                for (int i = 0; i < va.length; i++) {
                    out.putFloat(va[i] * vb[i]);
                }
            }
        }

        private static float[] values(RowSource src, long row, int rows, int inner) throws IOException {
            if (src.format().isBlock()) {
                throw new IllegalArgumentException(src.describe() + ": an Engram gate needs plain float values,"
                        + " got the block type " + src.format());
            }
            int elementBytes = (int) (src.rowBytes() / inner);
            ByteBuffer bytes = RowTransforms.readRows(src, row, rows);
            int total = Math.multiplyExact(rows, inner);
            float[] values = new float[total];
            for (int i = 0; i < total; i++) {
                DtypePolicy.decode(src.format(), bytes, i * elementBytes, values, i, 1);
            }
            return values;
        }
    }

    // ------------------------------------------------------------------ helpers

    private static void requireDtype(RowSource src, ElementFormat format, StDType expected, String what) {
        if (format.stDType() != expected) {
            throw new GgufFormatException(src.describe() + ": the Engram " + what + " has dtype " + format
                    + ", expected " + expected.headerName());
        }
    }

    private static float scale(int exponent) {
        return Float.intBitsToFloat(exponent << 23);
    }

    private static StDType outputDtype(GGMLType type) {
        return type == GGMLType.BF16 ? StDType.BF16 : StDType.F32;
    }

    private static long[] tableShape(List<Part> parts) {
        int dim = parts.get(0).dim();
        long rows = 0;
        for (Part p : parts) {
            if (p.dim() != dim) {
                throw new IllegalArgumentException("Engram tables of different widths: " + dim + " and " + p.dim());
            }
            rows = Math.addExact(rows, p.rows());
        }
        return new long[]{rows, dim};
    }

    private static String tableDescribe(List<Part> parts, GGMLType output) {
        StringBuilder sb = new StringBuilder("Engram table [");
        for (int i = 0; i < parts.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(parts.get(i).weight().describe());
        }
        return sb.append("] as ").append(output).toString();
    }
}
