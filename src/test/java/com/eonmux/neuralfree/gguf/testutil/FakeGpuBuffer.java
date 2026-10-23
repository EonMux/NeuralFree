package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import java.util.function.LongConsumer;

/**
 * Host-memory {@link GpuBuffer} used by {@link FakeComputeBackend}.
 *
 * <p>Contents are little-endian bytes, zero-initialised and allocated on first access, so buffers
 * that a test never inspects cost no heap. Content access is limited to buffers of at most
 * {@link #MAX_CONTENT_BYTES}; larger buffers can still be allocated, bound and recorded.
 *
 * <p>The read-only flag mirrors {@code allocateBuffer(size, true)}: it is the device-side access
 * hint the engine requested. Host-side access through this class is never restricted by it.
 *
 * <p>Element accessors take element indices (4 bytes for int/float, 2 bytes for short), not byte
 * offsets. Every access is bounds-checked and fails on a closed buffer.
 */
public final class FakeGpuBuffer implements GpuBuffer {

    /** Largest buffer whose contents can be read or written from the host. */
    public static final long MAX_CONTENT_BYTES = Integer.MAX_VALUE - 8L;

    private final long bufferId;
    private final long sizeInBytes;
    private final boolean readOnly;
    private final Object owner;
    private final LongConsumer releaseHook;
    private byte[] data;
    private boolean closed;

    /** Creates a standalone buffer that belongs to no backend. */
    public FakeGpuBuffer(long bufferId, long sizeInBytes, boolean readOnly) {
        this(bufferId, sizeInBytes, readOnly, null, null);
    }

    FakeGpuBuffer(long bufferId, long sizeInBytes, boolean readOnly, Object owner, LongConsumer releaseHook) {
        if (sizeInBytes <= 0) {
            throw new IllegalArgumentException("buffer size must be positive, got " + sizeInBytes);
        }
        this.bufferId = bufferId;
        this.sizeInBytes = sizeInBytes;
        this.readOnly = readOnly;
        this.owner = owner;
        this.releaseHook = releaseHook;
    }

    @Override
    public long sizeInBytes() {
        return sizeInBytes;
    }

    @Override
    public long bufferId() {
        return bufferId;
    }

    /** The device access hint requested at allocation. */
    public boolean isReadOnly() {
        return readOnly;
    }

    public synchronized boolean isClosed() {
        return closed;
    }

    Object owner() {
        return owner;
    }

    /** Releases the buffer. Idempotent; the owning backend's allocation total drops once. */
    @Override
    public void close() {
        boolean release;
        synchronized (this) {
            release = !closed;
            closed = true;
            data = null;
        }
        if (release && releaseHook != null) {
            releaseHook.accept(sizeInBytes);
        }
    }

    // ------------------------------------------------------------------ raw byte access

    /** Copies {@code len} bytes from {@code src[srcOff..]} to byte offset {@code offset}. */
    public synchronized void write(long offset, byte[] src, int srcOff, int len) {
        checkRange(offset, len);
        if (srcOff < 0 || len > src.length - srcOff) {
            throw new IndexOutOfBoundsException("source range [" + srcOff + ", " + srcOff + "+" + len
                    + ") exceeds array length " + src.length);
        }
        System.arraycopy(src, srcOff, storage(), (int) offset, len);
    }

    /** Copies {@code len} bytes at byte offset {@code offset} into {@code dst[dstOff..]}. */
    public synchronized void read(long offset, byte[] dst, int dstOff, int len) {
        checkRange(offset, len);
        if (dstOff < 0 || len > dst.length - dstOff) {
            throw new IndexOutOfBoundsException("destination range [" + dstOff + ", " + dstOff + "+" + len
                    + ") exceeds array length " + dst.length);
        }
        System.arraycopy(storage(), (int) offset, dst, dstOff, len);
    }

    /** Returns a copy of all bytes. */
    public synchronized byte[] toByteArray() {
        checkRange(0, sizeInBytes);
        return storage().clone();
    }

    /** Fills {@code [offset, offset+bytes)} with the little-endian 32-bit {@code pattern}. */
    public synchronized void fill(long offset, long bytes, int pattern) {
        if ((offset & 3) != 0 || (bytes & 3) != 0) {
            throw new IllegalArgumentException("fill offset " + offset + " and size " + bytes
                    + " must be multiples of 4");
        }
        checkRange(offset, bytes);
        byte[] s = storage();
        for (int p = (int) offset, end = (int) (offset + bytes); p < end; p += 4) {
            putIntLE(s, p, pattern);
        }
    }

    /** Device-to-device copy semantics; overlapping ranges of the same buffer are handled. */
    void copyFrom(FakeGpuBuffer src, long srcOffset, long dstOffset, long bytes) {
        if (bytes > MAX_CONTENT_BYTES) {
            throw new UnsupportedOperationException("copy of " + bytes + " bytes exceeds the host content limit");
        }
        if (src == this) {
            synchronized (this) {
                checkRange(srcOffset, bytes);
                checkRange(dstOffset, bytes);
                System.arraycopy(storage(), (int) srcOffset, storage(), (int) dstOffset, (int) bytes);
            }
            return;
        }
        byte[] tmp = new byte[(int) bytes];
        src.read(srcOffset, tmp, 0, (int) bytes);
        write(dstOffset, tmp, 0, (int) bytes);
    }

    // ------------------------------------------------------------------ element access

    public synchronized int getInt(long index) {
        checkRange(index * 4, 4);
        return getIntLE(storage(), (int) (index * 4));
    }

    public synchronized void putInt(long index, int value) {
        checkRange(index * 4, 4);
        putIntLE(storage(), (int) (index * 4), value);
    }

    public float getFloat(long index) {
        return Float.intBitsToFloat(getInt(index));
    }

    public void putFloat(long index, float value) {
        putInt(index, Float.floatToRawIntBits(value));
    }

    /** Reads the 16-bit element at {@code index}, e.g. an F16 KV entry, as raw bits. */
    public synchronized short getShort(long index) {
        checkRange(index * 2, 2);
        byte[] s = storage();
        int p = (int) (index * 2);
        return (short) ((s[p] & 0xFF) | (s[p + 1] << 8));
    }

    /** Writes the 16-bit element at {@code index} from raw bits, e.g. an F16 or BF16 value. */
    public synchronized void putShort(long index, short bits) {
        checkRange(index * 2, 2);
        byte[] s = storage();
        int p = (int) (index * 2);
        s[p] = (byte) bits;
        s[p + 1] = (byte) (bits >>> 8);
    }

    public synchronized int[] getInts(long index, int count) {
        checkRange(index * 4, count * 4L);
        byte[] s = storage();
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            out[i] = getIntLE(s, (int) ((index + i) * 4));
        }
        return out;
    }

    public float[] getFloats(long index, int count) {
        int[] bits = getInts(index, count);
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            out[i] = Float.intBitsToFloat(bits[i]);
        }
        return out;
    }

    public synchronized void putInts(long index, int... values) {
        checkRange(index * 4, values.length * 4L);
        byte[] s = storage();
        for (int i = 0; i < values.length; i++) {
            putIntLE(s, (int) ((index + i) * 4), values[i]);
        }
    }

    public void putFloats(long index, float... values) {
        int[] bits = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = Float.floatToRawIntBits(values[i]);
        }
        putInts(index, bits);
    }

    /** All contents as floats; the size must be a multiple of 4. */
    public float[] toFloatArray() {
        return getFloats(0, elementCount4());
    }

    /** All contents as ints; the size must be a multiple of 4. */
    public int[] toIntArray() {
        return getInts(0, elementCount4());
    }

    @Override
    public String toString() {
        return "buf#" + bufferId + "(" + sizeInBytes + "B" + (readOnly ? ",ro" : "") + ")";
    }

    // ------------------------------------------------------------------ internals

    private int elementCount4() {
        if ((sizeInBytes & 3) != 0) {
            throw new IllegalStateException(this + " size is not a multiple of 4 bytes");
        }
        return (int) Math.min(sizeInBytes / 4, Integer.MAX_VALUE);
    }

    private void checkRange(long offset, long len) {
        if (closed) {
            throw new IllegalStateException(this + " is closed");
        }
        if (offset < 0 || len < 0 || len > sizeInBytes - offset) {
            throw new IndexOutOfBoundsException("range [" + offset + ", " + offset + "+" + len + ") is outside "
                    + this);
        }
    }

    private byte[] storage() {
        if (data == null) {
            if (sizeInBytes > MAX_CONTENT_BYTES) {
                throw new UnsupportedOperationException(this + " exceeds the host content limit of "
                        + MAX_CONTENT_BYTES + " bytes");
            }
            data = new byte[(int) sizeInBytes];
        }
        return data;
    }

    static int getIntLE(byte[] s, int p) {
        return (s[p] & 0xFF) | (s[p + 1] & 0xFF) << 8 | (s[p + 2] & 0xFF) << 16 | (s[p + 3] & 0xFF) << 24;
    }

    static void putIntLE(byte[] s, int p, int v) {
        s[p] = (byte) v;
        s[p + 1] = (byte) (v >>> 8);
        s[p + 2] = (byte) (v >>> 16);
        s[p + 3] = (byte) (v >>> 24);
    }
}
