package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.RowQuantizers;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One output side of the quantize tool as a {@link ModelSource}: the tensor table with the decided
 * target types over the adjusted metadata, whose {@link WeightStore} produces every tensor's quantized
 * bytes on demand. A read of a byte range quantizes just the rows that range covers — the source rows of
 * the input store are decoded to F32 and quantized through {@link RowQuantizers} with the tensor's
 * importance-matrix slice, one row at a time, so no tensor is ever held in the heap whole, and
 * {@link com.eonmux.neuralfree.gguf.GgufWriter} streams the output through this store exactly as it
 * streams a stored file. From two threads on ({@code nthread} of the params) rows run in parallel; row
 * outputs are disjoint byte ranges, so the parallel form writes the same bytes.
 * <p>
 * The store closes nothing it did not open: the input store stays with the run that created it.
 */
final class QuantizedSource implements ModelSource {

    /** One output tensor: its decided type, its input place and its importance slice. */
    static final class Tensor {
        final String name;
        final long[] dims;
        final GGMLType dstType;
        final GGMLType srcType;
        /** Tensor data offset in the input store's coordinate space. */
        final long srcOffset;
        /** Tensor data offset in this source's virtual table. */
        final long virtualOffset;
        final int nPerRow;
        final long nRows;
        /** Rows of one expert matrix ({@code ne[1]}); the imatrix slice follows the expert index. */
        final long rowsPerExpert;
        final int rowBytesSrc;
        final int rowBytesDst;
        /** The importance values, {@code nPerRow} per expert matrix; null when the run has none. */
        final float[] weights;

        Tensor(String name, long[] dims, GGMLType dstType, GGMLType srcType, long srcOffset,
               long virtualOffset, int nPerRow, long nRows, long rowsPerExpert, int rowBytesSrc,
               int rowBytesDst, float[] weights) {
            this.name = Objects.requireNonNull(name, "name");
            this.dims = dims;
            this.dstType = dstType;
            this.srcType = srcType;
            this.srcOffset = srcOffset;
            this.virtualOffset = virtualOffset;
            this.nPerRow = nPerRow;
            this.nRows = nRows;
            this.rowsPerExpert = rowsPerExpert;
            this.rowBytesSrc = rowBytesSrc;
            this.rowBytesDst = rowBytesDst;
            this.weights = weights;
        }

        long sizeBytes() {
            return nRows * rowBytesDst;
        }
    }

    private final String filePath;
    private final Map<String, Object> metadata;
    private final Map<String, GGUFValueType> valueTypes;
    private final Map<String, GGUFValueType> arrayElementTypes;
    private final List<Tensor> tensors;
    private final WeightStore input;
    private final int nthread;

    QuantizedSource(String filePath, Map<String, Object> metadata, Map<String, GGUFValueType> valueTypes,
                    Map<String, GGUFValueType> arrayElementTypes, List<Tensor> tensors, WeightStore input,
                    int nthread) {
        this.filePath = Objects.requireNonNull(filePath, "filePath");
        this.metadata = Map.copyOf(metadata);
        this.valueTypes = Map.copyOf(valueTypes);
        this.arrayElementTypes = Map.copyOf(arrayElementTypes);
        this.tensors = List.copyOf(tensors);
        this.input = Objects.requireNonNull(input, "input");
        this.nthread = nthread;
    }

    /** The output tensors, in table order. */
    List<Tensor> tensors() {
        return tensors;
    }

    @Override
    public ModelFormat format() {
        return ModelFormat.GGUF;
    }

    @Override
    public String getFilePath() {
        return filePath;
    }

    @Override
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    @Override
    public GGUFValueType getValueType(String key) {
        return valueTypes.get(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        return arrayElementTypes.get(key);
    }

    @Override
    public String getString(String key) {
        Object v = metadata.get(key);
        return v instanceof String ? (String) v : null;
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        List<GGUFTensorInfo> out = new ArrayList<>(tensors.size());
        for (Tensor t : tensors) {
            out.add(new GGUFTensorInfo(t.name, t.dims, t.dstType, t.virtualOffset));
        }
        return out;
    }

    @Override
    public long getTensorCount() {
        return tensors.size();
    }

    @Override
    public WeightStore createMappedStore(StoreUse use) {
        return new Store();
    }

    @Override
    public TensorRegions openRegions(TensorReadMode mode) {
        throw new UnsupportedOperationException("a quantize output serves writes only");
    }

    @Override
    public void close() {
    }

    @Override
    public String toString() {
        return "quantize[" + tensors.size() + " tensors -> " + filePath + "]";
    }

    /**
     * The quantizing store: every read of the virtual table quantizes the covered rows from the input
     * store. One row of source bytes, one row of F32 and one row of encoded bytes are in memory at a time
     * per worker.
     */
    private final class Store implements WeightStore {

        private Tensor locate(long offset, int size) throws IOException {
            for (Tensor t : tensors) {
                long start = t.virtualOffset;
                if (offset >= start && offset + size <= start + t.sizeBytes()) {
                    return t;
                }
            }
            throw new IOException("quantize output '" + filePath + "': the byte range [" + offset + ", "
                    + (offset + size) + ") is outside every tensor of the table");
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            if (dstOffset < 0 || size < 0 || dstOffset + size > dst.length) {
                throw new IOException("quantize output '" + filePath + "': the destination range ["
                        + dstOffset + ", " + (dstOffset + size) + ") exceeds the array of " + dst.length);
            }
            final Tensor t = locate(tensorOffset, size);
            final long a = tensorOffset - t.virtualOffset;
            final long b = a + size;
            final long first = a / t.rowBytesDst;
            final long last = (b + t.rowBytesDst - 1) / t.rowBytesDst;
            final int rows = (int) (last - first);
            if (nthread >= 2 && rows > 1) {
                try {
                    CpuParallel.parallelFor(rows, 8, (from, to) -> {
                        byte[] srcBytes = new byte[t.rowBytesSrc];
                        float[] f32 = new float[t.nPerRow];
                        byte[] row = new byte[t.rowBytesDst];
                        for (int r = from; r < to; r++) {
                            try {
                                computeRow(t, first + r, srcBytes, f32, row);
                            } catch (IOException e) {
                                throw new java.io.UncheckedIOException(e);
                            }
                            copyRow(t, row, first + r, a, b, dst, dstOffset);
                        }
                    });
                } catch (java.io.UncheckedIOException e) {
                    throw e.getCause();
                }
            } else {
                byte[] srcBytes = new byte[t.rowBytesSrc];
                float[] f32 = new float[t.nPerRow];
                byte[] row = new byte[t.rowBytesDst];
                for (long r = first; r < last; r++) {
                    computeRow(t, r, srcBytes, f32, row);
                    copyRow(t, row, r, a, b, dst, dstOffset);
                }
            }
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            byte[] chunk = new byte[size];
            readTensor(tensorOffset, size, chunk, 0);
            dst.put(chunk);
        }

        @Override
        public void close() {
        }

        /** Decodes one source row to F32, checks it, and quantizes it into {@code row}. */
        private void computeRow(Tensor t, long rowIndex, byte[] srcBytes, float[] f32, byte[] row)
                throws IOException {
            input.readTensor(t.srcOffset + rowIndex * t.rowBytesSrc, t.rowBytesSrc, srcBytes, 0);
            CpuBlockDecoders.decodeElements(t.srcType,
                    ByteBuffer.wrap(srcBytes, 0, t.rowBytesSrc).order(ByteOrder.LITTLE_ENDIAN),
                    0L, f32, 0, t.nPerRow);
            for (int j = 0; j < t.nPerRow; j++) {
                if (!Float.isFinite(f32[j])) {
                    throw new IOException("tensor '" + t.name + "' row " + rowIndex + " holds the"
                            + " non-finite value " + f32[j] + "; the quantize tool refuses non-finite"
                            + " weights");
                }
            }
            int wOff = (int) ((rowIndex / t.rowsPerExpert) * (long) t.nPerRow);
            RowQuantizers.quantizeRow(t.dstType, f32, 0, row, 0, t.nPerRow, t.weights, wOff);
        }

        /** Copies the part of one computed row that the requested byte range covers. */
        private void copyRow(Tensor t, byte[] row, long rowIndex, long a, long b, byte[] dst, int dstOffset) {
            long rowStart = rowIndex * t.rowBytesDst;
            long from = Math.max(a, rowStart);
            long to = Math.min(b, rowStart + t.rowBytesDst);
            if (to > from) {
                System.arraycopy(row, (int) (from - rowStart), dst, (int) (dstOffset + (from - a)),
                        (int) (to - from));
            }
        }
    }
}
