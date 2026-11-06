package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.LongUnaryOperator;

/**
 * MLA's {@code attn_k_b [heads, kv_lora, nope]} as vLLM's {@code split_kv_b_proj} gives it to attention when the
 * quantized {@code kv_b_proj}'s scales do not run along the stored rows: {@code W_UK}, the dequantized
 * ({@code get_and_maybe_dequant_weights}, mla_attention.py:407-422) key rows of the quantized {@code kv_b}, in the
 * source's float type. vLLM's weight row {@code (h, j)} is column {@code j} of head {@code h}'s
 * {@code [kv_lora, nope]} block; it is quantized whole (row by row, or from the frozen scales of the group's scan
 * at its virtual row), decoded, and its values written back to the stored rows. Immutable and thread-safe; reads
 * whole heads, so not row-local.
 */
final class MlaKeyRows implements RowSource {

    /** One vLLM weight row quantized into the bytes of {@code type}. */
    interface RowQuantizer {
        void quantize(long vllmRow, float[] x, byte[] dst);
    }

    private final String ggufName;
    private final RowSource src;
    private final RowQuantizer quantizer;
    private final GGMLType type;
    private final float scale;
    private final long heads;
    private final int kvLora;
    private final int nope;
    private final LongUnaryOperator headStart;

    private MlaKeyRows(String ggufName, RowSource src, RowQuantizer quantizer, GGMLType type, float scale,
                       LongUnaryOperator headStart) {
        this.ggufName = Objects.requireNonNull(ggufName, "ggufName");
        this.src = Objects.requireNonNull(src, "src");
        this.kvLora = kvLoraOf(ggufName, src);
        this.heads = src.shape()[0];
        this.nope = (int) src.shape()[2];
        if (type == null || kvLora % type.getBlockSize() != 0) {
            throw new IllegalArgumentException(ggufName + ": a kv_b row of " + kvLora + " values is not a whole"
                    + " number of " + type + " blocks");
        }
        this.quantizer = quantizer;
        this.type = type;
        this.scale = scale;
        this.headStart = Objects.requireNonNull(headStart, "headStart");
    }

    /** The key rows of a {@code kv_b} quantized row by row by {@code quantizer}. */
    static MlaKeyRows rowLocal(String ggufName, RowSource src, OnlineQuantizer quantizer) {
        Objects.requireNonNull(quantizer, "quantizer");
        return new MlaKeyRows(ggufName, src, (row, x, dst) -> quantizer.quantizeRow(x, 0, x.length, dst, 0),
                quantizer.typeFor(kvLoraOf(ggufName, src)), 1.0f, h -> 0);
    }

    /**
     * The key rows of a {@code kv_b} quantized from the frozen scales of {@code frozen}, head {@code h}'s key rows
     * at the virtual rows from {@code at.row(h * kv_lora)} on.
     */
    static MlaKeyRows scanned(String ggufName, RowSource src, ScanningQuantizer frozen, FusedLayout.Placement at) {
        Objects.requireNonNull(frozen, "frozen");
        int kvLora = kvLoraOf(ggufName, src);
        float[] out = frozen.outputScale();
        if (out != null && out.length != 1) {
            throw new IllegalArgumentException(ggufName + ": kv_b has one output scale, got " + out.length);
        }
        return new MlaKeyRows(ggufName, src, (row, x, dst) -> frozen.quantizeRow(row, x, 0, x.length, dst, 0),
                frozen.typeFor(kvLora), out == null ? 1.0f : out[0], h -> at.row(h * kvLora));
    }

    /**
     * The {@code kv_lora} of {@code attn_k_b}'s float {@code [heads, kv_lora, nope]}: the row length of vLLM's
     * {@code kv_b}.
     *
     * @throws IllegalArgumentException naming the tensor for another shape or a quantized source
     */
    static int kvLoraOf(String ggufName, RowSource src) {
        long[] shape = src.shape();
        if (shape.length != 3 || !src.format().isFloat() || shape[1] <= 0 || shape[1] > Integer.MAX_VALUE
                || shape[2] <= 0 || shape[2] > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(ggufName + ": attn_k_b is a float [heads, kv_lora, nope], got "
                    + Arrays.toString(shape) + " " + src.format());
        }
        return (int) shape[1];
    }

    @Override
    public String describe() {
        return ggufName + " dequantized from kv_b quantized to " + type + " at load (from " + src.describe() + ")";
    }

    @Override
    public ElementFormat format() {
        return src.format();
    }

    @Override
    public long[] shape() {
        return src.shape();
    }

    @Override
    public long rows() {
        return src.rows();
    }

    @Override
    public long rowBytes() {
        return src.rowBytes();
    }

    @Override
    public boolean rowLocal() {
        return false;
    }

    @Override
    public void readRows(long first, int count, ByteBuffer dst) throws IOException {
        long end = first + count;
        if (first < 0 || count < 0 || end > rows()) {
            throw new IndexOutOfBoundsException(ggufName + ": rows [" + first + ", " + end + ") of " + rows());
        }
        for (long h = first / kvLora; h < heads && h * kvLora < end; h++) {
            float[][] w = head(h);
            long from = Math.max(first, h * kvLora);
            long to = Math.min(end, (h + 1) * kvLora);
            for (long r = from; r < to; r++) {
                FloatRows.encode(ggufName, src.format().ggmlType(), w[(int) (r - h * kvLora)], 0, nope, dst);
            }
        }
    }

    /** Head {@code h}'s stored rows {@code [kv_lora][nope]}, each value replaced by vLLM's dequantized one. */
    private float[][] head(long h) throws IOException {
        float[][] w = new float[kvLora][];
        FloatRows.consume(ggufName, src, nope, h * kvLora, kvLora, (r, x, off) ->
                w[(int) (r - h * kvLora)] = Arrays.copyOfRange(x, off, off + nope));
        float[] row = new float[kvLora];
        byte[] q = new byte[Math.toIntExact(type.getBytesForElements(kvLora))];
        for (int j = 0; j < nope; j++) {
            for (int c = 0; c < kvLora; c++) {
                row[c] = w[c][j];
            }
            quantizer.quantize(headStart.applyAsLong(h) + j, row, q);
            float[] d = CpuBlockDecoders.dequantize(q, type, kvLora);
            for (int c = 0; c < kvLora; c++) {
                w[c][j] = d[c] * scale;
            }
        }
        return w;
    }

    @Override
    public String toString() {
        return describe() + ", " + Arrays.toString(src.shape());
    }
}
