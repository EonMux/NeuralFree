package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.FusedLayout.Placement;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * The float source's rows, each row's stored floats dequantized to f32 and quantized into the target type's row
 * (llama-quant.cpp:273-283, the f32 repack the requantizer feeds its row quantizers from): row by row through an
 * {@link OnlineQuantizer}, or from the frozen scales of a {@link ScanningQuantizer} that
 * {@link RequantPolicy#quantizeFused} scanned at planning time, with this tensor's rows at their {@link Placement}
 * in the scanned tensor (a transposed member's rows as column runs, stored as the scanner's column type).
 * Immutable and thread-safe; row-local when the source is.
 */
final class QuantizedRows implements RowSource {

    private final String ggufName;
    private final RowSource src;
    private final OnlineQuantizer quantizer;
    private final ScanningQuantizer scanning;
    private final Placement placement;
    private final GGMLType type;
    private final float[] outputScale;
    private final ElementFormat format;
    private final int inner;
    private final long rows;

    private QuantizedRows(String ggufName, RowSource src, OnlineQuantizer quantizer, ScanningQuantizer frozen,
                          Placement placement, GGMLType type) {
        this.ggufName = Objects.requireNonNull(ggufName, "ggufName");
        this.src = Objects.requireNonNull(src, "src");
        this.inner = FusedLayout.innerOf(ggufName, src.shape());
        this.rows = src.rows();
        if (type == null || inner % type.getBlockSize() != 0) {
            throw new IllegalArgumentException(ggufName + ": a row of " + inner + " values is not a whole number of "
                    + type + " blocks" + (type == null ? "" : " of " + type.getBlockSize()));
        }
        this.type = type;
        this.format = ElementFormat.of(type);
        this.quantizer = quantizer;
        this.scanning = frozen;
        this.placement = placement;
        this.outputScale = frozen == null ? null : frozen.outputScale();
    }

    /** {@code src} quantized row by row by {@code quantizer}, stored as its {@link OnlineQuantizer#typeFor}. */
    static QuantizedRows rowLocal(String ggufName, RowSource src, OnlineQuantizer quantizer) {
        Objects.requireNonNull(quantizer, "quantizer");
        return new QuantizedRows(ggufName, src, quantizer, null, null,
                quantizer.typeFor(FusedLayout.innerOf(ggufName, src.shape())));
    }

    /**
     * {@code src} quantized from the scales of {@code frozen} at {@code placement}: a row member as the scanner's
     * {@link ScanningQuantizer#typeFor}, a transposed member as {@code columnType}.
     */
    static QuantizedRows scanned(String ggufName, RowSource src, ScanningQuantizer frozen, Placement placement,
                                 GGMLType columnType) {
        Objects.requireNonNull(frozen, "frozen");
        Objects.requireNonNull(placement, "placement");
        GGMLType t = placement.transposed() ? columnType
                : frozen.typeFor(FusedLayout.innerOf(ggufName, src.shape()));
        return new QuantizedRows(ggufName, src, null, frozen, placement, t);
    }

    @Override
    public String describe() {
        return ggufName + " quantized to " + type + " at load (from " + src.describe() + ")";
    }

    @Override
    public ElementFormat format() {
        return format;
    }

    @Override
    public long[] shape() {
        return src.shape();
    }

    @Override
    public long rows() {
        return rows;
    }

    @Override
    public long rowBytes() {
        return format.rowBytes(inner);
    }

    @Override
    public boolean rowLocal() {
        return src.rowLocal();
    }

    @Override
    public float[] outputScale() {
        return outputScale == null ? null : outputScale.clone();
    }

    @Override
    public void readRows(long first, int count, ByteBuffer dst) throws IOException {
        byte[] row = new byte[Math.toIntExact(rowBytes())];
        if (scanning == null) {
            OnlineQuantizer q = quantizer;
            FloatRows.consume(ggufName, src, inner, first, count, (r, x, off) -> {
                q.quantizeRow(x, off, inner, row, 0);
                dst.put(row);
            });
            return;
        }
        ScanningQuantizer s = scanning;
        Placement p = placement;
        GGMLType t = type;
        FloatRows.consume(ggufName, src, inner, first, count, (r, x, off) -> {
            if (p.transposed()) {
                s.quantizeColumn(p.row(r), p.column(r), x, off, inner, t, row, 0);
            } else {
                s.quantizeRow(p.row(r), x, off, inner, row, 0);
            }
            dst.put(row);
        });
    }

    @Override
    public String toString() {
        return describe() + ", " + Arrays.toString(src.shape());
    }
}
