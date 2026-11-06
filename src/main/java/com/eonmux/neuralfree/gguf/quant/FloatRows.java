package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** The stored float rows of a source, decoded to f32 chunk by chunk, and f32 rows encoded back to a float type. */
final class FloatRows {

    /** The scratch a chunk of rows keeps under, in bytes (per call). */
    private static final int CHUNK_BYTES = 1 << 20;

    private FloatRows() {
    }

    /** One decoded float row handed to a consumer with its row index and offset in the scratch. */
    interface RowSink {
        void row(long r, float[] x, int off) throws IOException;
    }

    /**
     * Reads rows {@code [first, first + count)} of {@code src}'s stored floats ({@code inner} per row), decodes each
     * to f32 and hands it to {@code sink}.
     */
    static void consume(String ggufName, RowSource src, int inner, long first, int count, RowSink sink)
            throws IOException {
        int inRowBytes = Math.toIntExact(src.rowBytes());
        float[] x = new float[inner];
        int chunk = Math.max(1, CHUNK_BYTES / Math.max(inRowBytes, 1));
        for (long r = first; r < first + count; r += chunk) {
            int rows = (int) Math.min(chunk, first + count - r);
            ByteBuffer in = ByteBuffer.allocate(rows * inRowBytes).order(ByteOrder.LITTLE_ENDIAN);
            src.readRows(r, rows, in);
            in.flip();
            for (int i = 0; i < rows; i++) {
                decode(ggufName, src.format().ggmlType(), in, inRowBytes, x);
                sink.row(r + i, x, 0);
            }
        }
    }

    /**
     * Writes the {@code n} floats at {@code x[off]} to {@code dst} as {@code t} (F32, F16 or BF16; round to nearest
     * even, as PyTorch's {@code Tensor.to}).
     */
    static void encode(String ggufName, GGMLType t, float[] x, int off, int n, ByteBuffer dst) {
        ByteBuffer out = dst.order() == ByteOrder.LITTLE_ENDIAN ? dst : dst.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = off; i < off + n; i++) {
            switch (t) {
                case F32:
                    out.putFloat(x[i]);
                    break;
                case F16:
                    out.putShort((short) FpBlockDecoders.fp32ToFp16(x[i]));
                    break;
                case BF16:
                    out.putShort((short) KvBlockCodec.fp32ToBf16(x[i]));
                    break;
                default:
                    throw new IllegalStateException(ggufName + ": no float encode to " + t);
            }
        }
        if (out != dst) {
            dst.position(out.position());
        }
    }

    /** Decodes one stored float row of {@code rowBytes} bytes of type {@code t} into {@code x}. */
    private static void decode(String ggufName, GGMLType t, ByteBuffer in, int rowBytes, float[] x) {
        int limit = in.position() + rowBytes;
        for (int i = 0; i < x.length; i++) {
            switch (t) {
                case F32:
                    x[i] = in.getFloat();
                    break;
                case F16:
                    x[i] = FpBlockDecoders.fp16ToFp32(in.getShort() & 0xFFFF);
                    break;
                case BF16:
                    x[i] = FpBlockDecoders.bf16ToFp32(in.getShort() & 0xFFFF);
                    break;
                case F64:
                    x[i] = (float) in.getDouble();
                    break;
                default:
                    throw new IllegalStateException(ggufName + ": no float decode of the source's " + t);
            }
        }
        if (in.position() != limit) {
            throw new IllegalStateException(ggufName + ": the stored float row read " + (in.position() - limit)
                    + " bytes short");
        }
    }
}
