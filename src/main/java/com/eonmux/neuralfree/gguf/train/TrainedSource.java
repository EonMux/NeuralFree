package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A {@link ModelSource} whose tensor bytes are the delegate's with the trained weights of a finetune run laid over
 * them: the view {@link com.eonmux.neuralfree.gguf.GgufWriter} exports after training, so the saved GGUF keeps the
 * model's metadata, tokenizer, quantization of every untouched tensor and the geometry of the file, and carries the
 * trained F32 values of the tensors the run stepped.
 *
 * <p>The tensor table is the delegate's: a trained tensor keeps its name, {@code ne} dims, place and stored type
 * ({@code F32}, {@code F16} or {@code BF16}), and only its bytes change. The float32 master values of the run are
 * encoded into the tensor's own stored type, so a F16 or BF16 model keeps its precision and size. The store serves a
 * read of any byte range mixed from the two — the trained bytes of the tensors named in the map, the delegate's bytes
 * everywhere else — so a chunked export that crosses a trained tensor's edge still reads the right bytes. The values
 * map is encoded once, at construction, so the source is independent of the trainer afterwards and immutable.</p>
 *
 * <p>Every name of the map must be a tensor of the delegate stored as {@link GGMLType#F32}, {@link GGMLType#F16} or
 * {@link GGMLType#BF16}, with one float32 value per element; anything else is refused by name, because a finetune run
 * only ever writes tensors it decoded as a trainable type and a name the file does not carry would be silently lost.
 * The source serves writes only: its {@link #openRegions(TensorReadMode)} is refused rather than serving the
 * delegate's untrained bytes. Closing it closes the delegate.</p>
 */
public final class TrainedSource implements ModelSource {

    private final ModelSource delegate;
    private final List<GGUFTensorInfo> tensors;
    private final Map<String, Trained> trained;

    /** One trained tensor: where its bytes sit in the virtual table and the bytes themselves. */
    private static final class Trained {
        private final long offset;
        private final int sizeBytes;
        private final byte[] bytes;

        Trained(long offset, byte[] bytes) {
            this.offset = offset;
            this.sizeBytes = bytes.length;
            this.bytes = bytes;
        }

        long end() {
            return offset + sizeBytes;
        }
    }

    private TrainedSource(ModelSource delegate, List<GGUFTensorInfo> tensors, Map<String, Trained> trained) {
        this.delegate = delegate;
        this.tensors = tensors;
        this.trained = trained;
    }

    /**
     * The trained view of {@code delegate}.
     *
     * @param values the trained float32 master values by stored tensor name
     * @throws IllegalArgumentException when a name is not a tensor of the model, is not stored as F32, F16 or BF16, or
     *                                  does not carry one value per element
     */
    public static TrainedSource of(ModelSource delegate, Map<String, float[]> values) {
        Objects.requireNonNull(delegate, "delegate");
        Objects.requireNonNull(values, "values");
        List<GGUFTensorInfo> table = List.copyOf(delegate.getTensors());
        Map<String, GGUFTensorInfo> byName = new HashMap<>();
        for (GGUFTensorInfo info : table) {
            byName.put(info.name(), info);
        }
        Map<String, Trained> trained = new LinkedHashMap<>();
        for (Map.Entry<String, float[]> e : values.entrySet()) {
            String name = e.getKey();
            GGUFTensorInfo info = name == null ? null : byName.get(name);
            if (info == null) {
                throw new IllegalArgumentException("the model has no tensor '" + name + "'; the trained values of "
                        + "a finetune run must name stored tensors");
            }
            if (!isTrainableType(info.type())) {
                throw new IllegalArgumentException("tensor '" + name + "' is stored as " + info.type()
                        + ", not F32, F16 or BF16; a finetune run writes those types only");
            }
            float[] data = Objects.requireNonNull(e.getValue(), "values of '" + name + "'");
            if (data.length != info.numElements()) {
                throw new IllegalArgumentException("tensor '" + name + "' holds " + info.numElements()
                        + " elements, but the trained values carry " + data.length);
            }
            trained.put(name, new Trained(info.offset(), toBytes(name, info.type(), data)));
        }
        return new TrainedSource(delegate, table, Collections.unmodifiableMap(trained));
    }

    /** Whether a finetune run stores the tensor's trained values back as this type. */
    private static boolean isTrainableType(GGMLType type) {
        return type == GGMLType.F32 || type == GGMLType.F16 || type == GGMLType.BF16;
    }

    /**
     * The trained bytes of one tensor in its stored type, little-endian as GGUF stores it: the float32 master values
     * themselves for F32, rounded to the nearest F16 or BF16 for those types.
     */
    private static byte[] toBytes(String name, GGMLType type, float[] data) {
        if (type == GGMLType.F32) {
            ByteBuffer buffer = ByteBuffer.allocate(data.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            buffer.asFloatBuffer().put(data);
            return buffer.array();
        }
        int elementBytes = type.getTypeSize();
        if (elementBytes != Short.BYTES) {
            throw new IllegalStateException("unreachable trainable type " + type + " for tensor '" + name + "'");
        }
        ByteBuffer buffer = ByteBuffer.allocate(data.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : data) {
            int bits = type == GGMLType.F16 ? HalfFloats.floatBitsToHalf(Float.floatToRawIntBits(value))
                    : HalfFloats.floatBitsToBFloat16(Float.floatToRawIntBits(value));
            buffer.putShort((short) bits);
        }
        return buffer.array();
    }

    /** The number of tensors whose bytes the view replaces. */
    public int trainedTensorCount() {
        return trained.size();
    }

    @Override
    public ModelFormat format() {
        return delegate.format();
    }

    @Override
    public String getFilePath() {
        return delegate.getFilePath();
    }

    @Override
    public Map<String, Object> getMetadata() {
        return delegate.getMetadata();
    }

    @Override
    public GGUFValueType getValueType(String key) {
        return delegate.getValueType(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        return delegate.getArrayElementType(key);
    }

    @Override
    public String getString(String key) {
        return delegate.getString(key);
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        return tensors;
    }

    @Override
    public long getTensorCount() {
        return delegate.getTensorCount();
    }

    @Override
    public WeightStore createMappedStore(StoreUse use) throws IOException {
        WeightStore base = delegate.createMappedStore(use);
        try {
            return new TrainedStore(base);
        } catch (RuntimeException e) {
            base.close();
            throw e;
        }
    }

    /** The trained view serves the export's streaming reads; a host-side region reader would see the delegate. */
    @Override
    public TensorRegions openRegions(TensorReadMode mode) {
        throw new UnsupportedOperationException("a trained source serves the export's tensor reads only; its"
                + " region view would serve the untrained bytes of '" + getFilePath() + "'");
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public String toString() {
        return "trained[" + trained.size() + " tensors] over " + delegate;
    }

    /**
     * The store of the view: every read of a byte range is served from the trained bytes of the tensors that cover
     * it and from the delegate everywhere else. The delegate store belongs to this store and closes with it.
     */
    private final class TrainedStore implements WeightStore {

        private final WeightStore base;
        private final List<Trained> ranges;

        TrainedStore(WeightStore base) {
            this.base = base;
            List<Trained> ordered = new ArrayList<>(trained.values());
            ordered.sort((a, b) -> Long.compare(a.offset, b.offset));
            this.ranges = ordered;
        }

        /** The trained tensor covering {@code offset}, or null. */
        private Trained covering(long offset) {
            int low = 0;
            int high = ranges.size() - 1;
            while (low <= high) {
                int mid = (low + high) >>> 1;
                Trained t = ranges.get(mid);
                if (offset < t.offset) {
                    high = mid - 1;
                } else if (offset >= t.end()) {
                    low = mid + 1;
                } else {
                    return t;
                }
            }
            return null;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            if (tensorOffset < 0 || size < 0 || dstOffset < 0 || dstOffset + size > dst.length) {
                throw new IOException("trained source of '" + getFilePath() + "': the read [" + tensorOffset
                        + ", " + (tensorOffset + size) + ") into [" + dstOffset + ", " + (dstOffset + size)
                        + ") does not fit the destination array of " + dst.length);
            }
            int done = 0;
            while (done < size) {
                long at = tensorOffset + done;
                Trained t = covering(at);
                if (t == null) {
                    base.readTensor(at, size - done, dst, dstOffset + done);
                    return;
                }
                int chunk = (int) Math.min(size - done, t.end() - at);
                System.arraycopy(t.bytes, (int) (at - t.offset), dst, dstOffset + done, chunk);
                done += chunk;
            }
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            byte[] bytes = new byte[size];
            readTensor(tensorOffset, size, bytes, 0);
            dst.put(bytes);
        }

        @Override
        public void close() throws IOException {
            base.close();
        }
    }
}
