package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The dummy load ({@code --load-format dummy}): a {@link ModelSource} of the delegate source's metadata, tensor
 * table and file path whose tensor bytes are seeded random weights instead of a file's (llama.cpp-compatible
 * benchmark setups; vLLM and SGLang dummy loaders). Consumers read the weights through the same seams as any
 * source: the {@link WeightStore} of {@code createMappedStore} and the regions of {@code openRegions}, with the
 * declared tensor types. A dummy load has no content inputs, so a model loaded this way contributes nothing to
 * the weights digest and has nothing to persist.
 *
 * <p>Every element decodes to a finite value of its declared type: F32, F64, F16, BF16 encode a seeded uniform
 * value directly, the 8 / 16 / 32 / 64-bit integer types encode seeded integers, and every remaining (block
 * quantized or 8-bit float) type takes a seeded block whose decoded values are checked with
 * {@link CpuBlockDecoders}; a block with a NaN or infinite value is written as zeros instead. The bytes of a
 * block depend only on the block's tensor name and block index, so any read granularity of a tensor, through the
 * store or the regions, yields the same bytes. Immutable beyond the delegate; thread-safe.
 */
public final class DummyWeights implements ModelSource {

    private final ModelSource base;
    /** Guards a second {@link #close()}; true after the first close. */
    private final java.util.concurrent.atomic.AtomicBoolean closed = new java.util.concurrent.atomic.AtomicBoolean();
    private final GGUFTensorInfo[] byOffset;

    private DummyWeights(ModelSource base, GGUFTensorInfo[] byOffset) {
        this.base = base;
        this.byOffset = byOffset;
    }

    /**
     * The dummy view of {@code base}; the view owns it and {@link #close()} closes the delegate.
     *
     * @throws IllegalArgumentException when two tensors of {@code base} overlap in virtual bytes, or a tensor's
     *                                 storage size is not a whole number of its storage blocks
     */
    public static DummyWeights over(ModelSource base) {
        Objects.requireNonNull(base, "base");
        GGUFTensorInfo[] sorted = base.getTensors().toArray(new GGUFTensorInfo[0]);
        List<GGUFTensorInfo> view = new ArrayList<>();
        for (GGUFTensorInfo info : sorted) {
            Objects.requireNonNull(info, "tensor");
            if (info.sizeInBytes() % Math.max(1L, (long) info.type().getTypeSize()) != 0) {
                throw new IllegalArgumentException("tensor '" + info.name() + "' of " + base.getFilePath()
                        + " stores " + info.sizeInBytes() + " bytes, which is not a whole number of "
                        + info.type() + " blocks (" + info.type().getTypeSize() + " bytes)");
            }
            view.add(info);
        }
        view.sort((a, b) -> Long.compare(a.offset(), b.offset()));
        GGUFTensorInfo[] byOffset = view.toArray(new GGUFTensorInfo[0]);
        for (int i = 1; i < byOffset.length; i++) {
            GGUFTensorInfo prev = byOffset[i - 1];
            if (byOffset[i].offset() < prev.offset() + prev.sizeInBytes()) {
                throw new IllegalArgumentException("tensor '" + byOffset[i].name() + "' of " + base.getFilePath()
                        + " (" + byOffset[i].offset() + ".." + (byOffset[i].offset() + byOffset[i].sizeInBytes())
                        + ") overlaps tensor '" + prev.name() + "' (" + prev.offset() + ".."
                        + (prev.offset() + prev.sizeInBytes()) + ")");
            }
        }
        return new DummyWeights(base, byOffset);
    }

    private GGUFTensorInfo tensorAt(long offset) {
        int lo = 0;
        int hi = byOffset.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (byOffset[mid].offset() <= offset) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        GGUFTensorInfo info = byOffset[lo];
        if (offset < info.offset() || offset > info.offset() + info.sizeInBytes()) {
            throw new IllegalArgumentException("no tensor of " + base.getFilePath() + " holds virtual byte offset "
                    + offset);
        }
        return info;
    }

    @Override
    public ModelFormat format() {
        return base.format();
    }

    @Override
    public String getFilePath() {
        return base.getFilePath();
    }

    @Override
    public Map<String, Object> getMetadata() {
        return base.getMetadata();
    }

    @Override
    public com.eonmux.neuralfree.gguf.GGUFValueType getValueType(String key) {
        return base.getValueType(key);
    }

    @Override
    public com.eonmux.neuralfree.gguf.GGUFValueType getArrayElementType(String key) {
        return base.getArrayElementType(key);
    }

    @Override
    public String getString(String key) {
        return base.getString(key);
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        return base.getTensors();
    }

    @Override
    public long getTensorCount() {
        return base.getTensorCount();
    }

    @Override
    public WeightStore createMappedStore(StoreUse use) {
        return new DummyStore();
    }

    @Override
    public WeightStore createMappedStore() {
        return createMappedStore(StoreUse.RESIDENT_UPLOAD);
    }

    @Override
    public TensorRegions openRegions(TensorReadMode mode) {
        Objects.requireNonNull(mode, "mode");
        return new DummyRegions();
    }

    @Override
    public void close() throws IOException {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        // the view owns the delegate's lifetime; a second close does nothing
        base.close();
    }

    @Override
    public String toString() {
        return "dummy " + base.getFilePath();
    }

    // ------------------------------------------------------------------ consumers

    /** The dummy {@link WeightStore}: tensor reads generate the seeded bytes of the requested virtual range. */
    private final class DummyStore implements WeightStore {

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            if (size < 0) {
                throw new IOException("cannot read " + size + " bytes of dummy weights at " + tensorOffset);
            }
            if (dstOffset + (long) size > dst.length) {
                throw new IOException("cannot read " + size + " bytes of dummy weights at " + tensorOffset
                        + " into an array of " + dst.length + " bytes at " + dstOffset);
            }
            GGUFTensorInfo info = tensorAt(tensorOffset);
            long rel = tensorOffset - info.offset();
            if (rel + (long) size > info.sizeInBytes()) {
                throw new IOException("dummy weights of '" + info.name() + "' of " + DummyWeights.this
                        .getFilePath() + " end at its " + info.sizeInBytes() + " bytes; a read of " + size
                        + " bytes at " + rel + " exceeds it");
            }
            generate(info, rel, dst, dstOffset, size);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            byte[] buf = new byte[size];
            readTensor(tensorOffset, size, buf, 0);
            dst.put(buf);
        }

        @Override
        public void close() {
            // the dummy weights own no files
        }
    }

    /** The dummy {@link TensorRegions}: regions generate the seeded bytes of the requested tensor range. */
    private final class DummyRegions implements TensorRegions {

        @Override
        public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
            Objects.requireNonNull(info, "info");
            if (byteOff < 0 || length < 0 || length > Integer.MAX_VALUE
                    || byteOff > Long.MAX_VALUE - length || byteOff + length > info.sizeInBytes()) {
                throw new IOException("tensor " + info.name() + ": region [" + byteOff + ", +" + length
                        + ") is not a valid byte range of the dummy load of " + DummyWeights.this.getFilePath());
            }
            byte[] buf = new byte[(int) length];
            generate(info, byteOff, buf, 0, buf.length);
            return ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);
        }

        @Override
        public String description() {
            return "dummy " + base.getFilePath();
        }

        @Override
        public void close() {
            // the dummy regions own no files
        }
    }

    // ------------------------------------------------------------------ the seeded weights

    /**
     * Fills {@code buf[from .. from+len)} with the seeded weights of {@code info} starting {@code byteOff} bytes
     * into the tensor. Blocks are generated whole, so any granularity and order of reads yields the same bytes.
     */
    private static void generate(GGUFTensorInfo info, long byteOff, byte[] buf, int from, int len) {
        int typeSize = info.type().getTypeSize();
        long pos = 0;
        while (pos < len) {
            long blockStart = byteOff + pos;
            long blockIndex = blockStart / typeSize;
            int blockOffset = (int) (blockStart - blockIndex * typeSize);
            int available = Math.min(typeSize - blockOffset, len - (int) pos);
            byte[] block = new byte[typeSize];
            fillBlock(info, block, blockIndex);
            System.arraycopy(block, blockOffset, buf, from + (int) pos, available);
            pos += available;
        }
    }

    /** The bytes of one stored block of {@code info} at block index {@code blockIndex}. */
    private static void fillBlock(GGUFTensorInfo info, byte[] block, long blockIndex) {
        GGMLType type = info.type();
        int typeSize = type.getTypeSize();
        SplitMix64 rng = new SplitMix64(nameSeed(info) ^ mix64(blockIndex + 1));
        if (!type.isQuantized()) {
            switch (type) {
                case F32:
                    for (int o = 0; o < typeSize; o += 4) {
                        putIntLE(block, o, Float.floatToRawIntBits(uniformFloat(rng)));
                    }
                    return;
                case F64:
                    for (int o = 0; o < typeSize; o += 8) {
                        putLongLE(block, o, uniformDoubleBits(rng));
                    }
                    return;
                case F16:
                    for (int o = 0; o < typeSize; o += 2) {
                        putShortLE(block, o, (short) HalfFloats.floatToHalf(uniformFloat(rng)));
                    }
                    return;
                case BF16:
                    for (int o = 0; o < typeSize; o += 2) {
                        putShortLE(block, o, (short) HalfFloats.floatToBFloat16(uniformFloat(rng)));
                    }
                    return;
                case I8:
                    rng.nextBytes(block, 0, typeSize);
                    return;
                case I16:
                    for (int o = 0; o < typeSize; o += 2) {
                        putShortLE(block, o, (short) (rng.next() >>> 48));
                    }
                    return;
                case I32:
                    for (int o = 0; o < typeSize; o += 4) {
                        putIntLE(block, o, (int) mix64(rng.next()));
                    }
                    return;
                case I64:
                    for (int o = 0; o < typeSize; o += 8) {
                        putLongLE(block, o, rng.next());
                    }
                    return;
                default:
                    // every other non-quantized element type takes checked random bytes below
            }
        }
        rng.nextBytes(block, 0, typeSize);
        int blockElements = type.getBlockSize();
        float[] decoded = new float[blockElements];
        CpuBlockDecoders.decode(type, ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN), 0L, decoded, 0, 1);
        for (int i = 0; i < blockElements; i++) {
            float v = decoded[i];
            if (v != v || v == Float.NEGATIVE_INFINITY || v == Float.POSITIVE_INFINITY) {
                Arrays.fill(block, (byte) 0);
                return;
            }
        }
    }

    /** A uniform float of 2^-24 granularity in [-1, 1). */
    private static float uniformFloat(SplitMix64 rng) {
        long bits = rng.next() >>> 24;
        return (float) ((bits >> 1) - 0x800000L) * (1.0f / 8388608.0f);
    }

    /** The bits of a uniform double of 2^-52 granularity in [-1, 1). */
    private static long uniformDoubleBits(SplitMix64 rng) {
        long r = rng.next();
        double d = (r >>> 12) * 0x1p-51 - 1.0;
        return Double.doubleToRawLongBits(d);
    }

    private static long nameSeed(GGUFTensorInfo info) {
        long seed = 0xcbf29ce484222325L;
        for (byte b : info.name().getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
            seed = (seed ^ (b & 0xffL)) * 0x100000001b3L;
        }
        return seed ^ mix64(info.type().getId() * 2 + 1);
    }

    private static long mix64(long x) {
        x ^= x >>> 33;
        x *= 0xff51afd7ed558ccdL;
        x ^= x >>> 33;
        x *= 0xc4ceb9fe1a85ec53L;
        x ^= x >>> 33;
        return x;
    }

    private static void putShortLE(byte[] b, int off, short v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
    }

    private static void putIntLE(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >>> 8);
        b[off + 2] = (byte) (v >>> 16);
        b[off + 3] = (byte) (v >>> 24);
    }

    private static void putLongLE(byte[] b, int off, long v) {
        for (int i = 0; i < 8; i++) {
            b[off + i] = (byte) (v >>> (i * 8));
        }
    }

    /** The 64-bit stream the seeded weights draw from, one tensor block's stream per {@link SplitMix64}. */
    private static final class SplitMix64 {
        private long state;

        SplitMix64(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9e3779b97f4a7c15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
            z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
            return z ^ (z >>> 31);
        }

        /** Fills {@code out[from .. from+len)} with the stream's bytes, 8 at a time, little-endian. */
        void nextBytes(byte[] out, int from, int len) {
            int o = from;
            while (o + 8 <= from + len) {
                putLongLE(out, o, next());
                o += 8;
            }
            long tail = next();
            for (int i = 0; o < from + len; i++, o++) {
                out[o] = (byte) (tail >>> (i * 8));
            }
        }
    }
}
