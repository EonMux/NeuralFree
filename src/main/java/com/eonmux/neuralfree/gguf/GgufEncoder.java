package com.eonmux.neuralfree.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * The single GGUF encoder of the repository: it writes the little-endian bytes of a GGUF v3 file in the
 * field order of GGUF v2/v3 shared with llama.cpp (ggml/src/gguf.cpp and gguf-py/gguf/gguf_writer.py) —
 * magic, version, tensor count, key/value count, the key/value section, the tensor infos, zero padding to
 * the alignment, then each tensor's data, also padded. Every other writer of the repository routes its
 * bytes through this class, so test fixtures and exported models are byte-compatible.
 *
 * <p>Values are the same decoded Java representation {@link GGUFFileParser} reads back: {@link Integer}
 * for the 8/16/32-bit integer types (UINT32 and UINT64 keeping their bit patterns), {@link Long} for the
 * 64-bit integer types, {@link Float}, {@link Double}, {@link Boolean}, {@link String} and, for
 * {@link GGUFValueType#ARRAY}, a {@code List<Object>} of the element values in file order with their
 * element type given separately. Types are explicit, so UINT32 and INT32 share a Java type but not an id.
 *
 * <p>The encoder validates its input with named errors: an empty key, a key or tensor name longer than the
 * GGUF maximum, a nested array, a value of the wrong Java type, an out-of-range element or an entry count,
 * string length or array count beyond a limit is reported with the key or tensor in the message and nothing
 * is written. Malformed GGUF files that only the test fixtures need, such as a version other than 3, are
 * produced by editing the encoder's raw output bytes afterwards. Not thread-safe; buffered through a
 * private buffer, so the channel position of the underlying file is free for the caller to use.</p>
 */
public final class GgufEncoder {

    /** File magic ({@code GGUF_MAGIC} and {@link GGUFFileParser#MAGIC}). */
    public static final String MAGIC = "GGUF";

    /** GGUF version written ({@code GGUF_VERSION}, ggml/include/gguf.h:42). */
    public static final int GGUF_VERSION = 3;

    /** Longest string in bytes ({@code GGUF_MAX_STRING_LENGTH}, the read side's {@link GGUFHeaderReader}). */
    public static final long MAX_STRING_LENGTH = 1L << 30;

    /** Largest array element count ({@code GGUF_MAX_ARRAY_ELEMENTS}, gguf.cpp:21). */
    public static final long MAX_ARRAY_ELEMENTS = 1L << 30;

    private static final byte[] MAGIC_BYTES = {'G', 'G', 'U', 'F'};

    private static final int BUFFER_BYTES = 1 << 16;

    private final String path;
    private final WritableByteChannel out;
    private final ByteBuffer buf;
    private long position;

    /**
     * An encoder over {@code out}; {@code path} names the file being written, for close-overwrites in
     * channel messages and for {@link #checkName}. The caller owns the channel and closes it after the
     * encoder's buffered bytes are flushed with {@link #flush()}.
     */
    public GgufEncoder(WritableByteChannel out, String path) {
        if (out == null) {
            throw new IllegalArgumentException("output channel is null");
        }
        this.path = path == null ? "the GGUF file" : path;
        this.out = out;
        this.buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** File offset of the next byte written; end of the header once the last tensor info went out. */
    public long position() {
        return position;
    }

    /**
     * Writes magic, version, tensor count and key/value count. The channel is this encoder's own unless a
     * different one is passed, which must then own the position bookkeeping of the whole file.
     */
    public void writeHeader(WritableByteChannel out, long tensorCount, long kvCount) throws IOException {
        checkCounts(tensorCount, kvCount);
        writeBytes(MAGIC_BYTES, 0, MAGIC_BYTES.length);
        sink().putInt(GGUF_VERSION);
        sink().putLong(tensorCount);
        sink().putLong(kvCount);
        flush();
    }

    /**
     * Writes one key/value pair. {@code elementType} is required for {@link GGUFValueType#ARRAY} and must
     * not be an array itself, mirroring the reader's rule against nested arrays (gguf.cpp:606-611); its
     * value is the decoded representation described on the class.
     *
     * @throws IOException when the key is empty, a string or array length exceeds a limit, the pair is
     *                     malformed, or the value type of any element contradicts {@code type}
     */
    public void writeKv(String key, GGUFValueType type, GGUFValueType elementType, Object value)
            throws IOException {
        if (key == null || key.isEmpty()) {
            throw new IOException("a GGUF metadata key must be non-empty");
        }
        byte[] keyBytes = utf8(key, "key '" + key + "'");
        checkName(keyBytes, key);
        if (type == null) {
            throw fail(key, "its GGUF value type is missing");
        }
        if (type == GGUFValueType.ARRAY) {
            if (elementType == null) {
                throw fail(key, "an ARRAY value needs an element type");
            }
            if (elementType == GGUFValueType.ARRAY) {
                throw fail(key, GGUFValueType.ARRAY + " values of GGUF arrays are invalid");
            }
            if (!(value instanceof List)) {
                throw fail(key, ", an " + GGUFValueType.ARRAY + " value, is not a list of values");
            }
        } else if (elementType != null) {
            throw fail(key, "takes no element type but one is given: " + elementType);
        }
        put(keyBytes);
        putU32(type.id());
        if (type == GGUFValueType.ARRAY) {
            putArray(key, elementType, (List<?>) value);
        } else {
            putScalar(key, type, value);
        }
    }

    /**
     * Writes one tensor info: name, {@code n_dims}, the {@code ne[]} extents, ggml type id and the data
     * offset relative to the data section. The structural checks of
     * {@link GGUFTensorInfo#validate(String, GGUFTensorInfo)} are not repeated here; the caller runs them.
     */
    public void writeTensorInfo(GGUFTensorInfo info, long relOffset) throws IOException {
        if (info == null) {
            throw new IOException("tensor info is null");
        }
        if (relOffset < 0) {
            throw new IOException("tensor '" + info.name() + "' has a negative data offset " + relOffset);
        }
        byte[] name = checkName(utf8(info.name(), "name of tensor '" + info.name() + "'"), info.name());
        long[] dims = info.dims();
        if (dims == null || dims.length == 0 || dims.length > GGUFTensorInfo.MAX_DIMS) {
            throw new IOException("tensor '" + info.name() + "' needs 1 to " + GGUFTensorInfo.MAX_DIMS
                    + " dimensions");
        }
        if (info.type() == null) {
            throw new IOException("tensor '" + info.name() + "' has no ggml type");
        }
        put(name);
        putU32(dims.length);
        for (long dim : dims) {
            sink().putLong(dim);
        }
        putU32(info.type().getId());
        sink().putLong(relOffset);
    }

    /** Writes {@code data[off, off + len)} of tensor data. */
    public void writeData(byte[] data, int off, int len) throws IOException {
        if (data == null) {
            throw new IllegalArgumentException("tensor data is null");
        }
        if (off < 0 || len < 0 || len > data.length - off) {
            throw new IllegalArgumentException("data range [" + off + ", +" + len + ") exceeds array length "
                    + data.length);
        }
        writeBytes(data, off, len);
    }

    /** Writes {@code data.length} bytes of tensor data. */
    public void writeData(byte[] data) throws IOException {
        writeData(data, 0, data.length);
    }

    /**
     * Pads the file with zero bytes to the next multiple of {@code alignment}, as GGUF pads the header
     * and every tensor (gguf.cpp:649-655).
     */
    public void align(int alignment) throws IOException {
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IOException("general.alignment " + alignment + " is not a positive power of two");
        }
        long pad = (alignment - position % alignment) % alignment;
        byte[] zeros = new byte[(int) Math.min(pad, 4096)];
        while (pad > 0) {
            int n = (int) Math.min(pad, zeros.length);
            writeBytes(zeros, 0, n);
            pad -= n;
        }
    }

    /** Passes the buffered bytes to the channel; the last call before the caller closes it. */
    public void flush() throws IOException {
        buf.flip();
        try {
            writeRemaining();
        } finally {
            buf.clear();
        }
    }

    // ------------------------------------------------------------------ key/value values

    private void putScalar(String key, GGUFValueType type, Object v) throws IOException {
        switch (type) {
            case UINT8: {
                Integer n = integerValue(key, type, v);
                if (n < 0 || n > 0xFF) {
                    throw outOfRange(key, type, n, 0, 0xFF);
                }
                putU8(n);
                break;
            }
            case INT8: {
                Integer n = integerValue(key, type, v);
                if (n < Byte.MIN_VALUE || n > Byte.MAX_VALUE) {
                    throw outOfRange(key, type, n, Byte.MIN_VALUE, Byte.MAX_VALUE);
                }
                putU8(n);
                break;
            }
            case UINT16: {
                Integer n = integerValue(key, type, v);
                if (n < 0 || n > 0xFFFF) {
                    throw outOfRange(key, type, n, 0, 0xFFFF);
                }
                putU16(n);
                break;
            }
            case INT16: {
                Integer n = integerValue(key, type, v);
                if (n < Short.MIN_VALUE || n > Short.MAX_VALUE) {
                    throw outOfRange(key, type, n, Short.MIN_VALUE, Short.MAX_VALUE);
                }
                putU16(n);
                break;
            }
            case UINT32:
            case INT32:
                sink().putInt(integerValue(key, type, v));
                break;
            case FLOAT32:
                if (!(v instanceof Float)) {
                    throw notA(key, type, v, "a FLOAT32 value");
                }
                sink().putInt(Float.floatToRawIntBits((Float) v));
                break;
            case BOOL:
                if (!(v instanceof Boolean)) {
                    throw notA(key, type, v, "a BOOL value");
                }
                putU8((Boolean) v ? 1 : 0);
                break;
            case STRING: {
                if (!(v instanceof String)) {
                    throw notA(key, type, v, "a STRING value");
                }
                byte[] bytes = utf8((String) v, "value of key '" + key + "'");
                sink().putLong(bytes.length);
                put(bytes);
                break;
            }
            case UINT64:
            case INT64:
                if (!(v instanceof Long)) {
                    throw notA(key, type, v, "a " + type + " value (a JVM long)");
                }
                sink().putLong((Long) v);
                break;
            case FLOAT64:
                if (!(v instanceof Double)) {
                    throw notA(key, type, v, "a FLOAT64 value");
                }
                sink().putLong(Double.doubleToRawLongBits((Double) v));
                break;
            default:
                throw fail(key, ", an " + GGUFValueType.ARRAY + " value, was written as a scalar");
        }
    }

    private void putArray(String key, GGUFValueType elementType, List<?> values) throws IOException {
        putU32(elementType.id());
        long count = values.size();
        if (count > MAX_ARRAY_ELEMENTS) {
            throw new IOException("array key '" + key + "' has " + count + " elements, more than the maximum of "
                    + MAX_ARRAY_ELEMENTS);
        }
        sink().putLong(count);
        for (int i = 0; i < values.size(); i++) {
            Object v = values.get(i);
            if (v == null) {
                throw fail(element(key, i), "is null");
            }
            putScalar(element(key, i), elementType, v);
        }
    }

    private Integer integerValue(String key, GGUFValueType type, Object v) throws IOException {
        if (!(v instanceof Integer)) {
            throw notA(key, type, v, "a " + type + " value (a JVM int)");
        }
        return (Integer) v;
    }

    private static String element(String key, int index) {
        return key + "[" + index + "]";
    }

    private IOException fail(String key, String detail) {
        return new IOException("metadata key '" + key + "'" + detail);
    }

    private IOException outOfRange(String key, GGUFValueType type, int v, int lo, int hi) {
        return new IOException("metadata key '" + key + "' value " + v + " of type " + type
                + " is outside [" + lo + ", " + hi + "]");
    }

    private IOException notA(String key, GGUFValueType type, Object value, String expected) {
        return new IOException("metadata key '" + key + "' of type " + type + " needs " + expected
                + " but gives " + (value == null ? "null" : value.getClass().getName()));
    }

    // ------------------------------------------------------------------ byte sink

    private ByteBuffer sink() throws IOException {
        if (buf.remaining() < 8) {
            flush();
        }
        return buf;
    }

    private void putU32(int v) throws IOException {
        if (buf.remaining() < 4) {
            flush();
        }
        buf.putInt(v);
        position += 4;
    }

    private void putU16(int v) throws IOException {
        if (buf.remaining() < 2) {
            flush();
        }
        buf.putShort((short) v);
        position += 2;
    }

    private void putU8(int v) throws IOException {
        if (buf.remaining() < 1) {
            flush();
        }
        buf.put((byte) v);
        position += 1;
    }

    private void put(byte[] bytes) throws IOException {
        writeBytes(bytes, 0, bytes.length);
    }

    private void writeBytes(byte[] bytes, int off, int len) throws IOException {
        position += len;
        int i = off;
        while (len > 0) {
            if (!buf.hasRemaining()) {
                flush();
            }
            int n = Math.min(len, buf.remaining());
            buf.put(bytes, i, n);
            i += n;
            len -= n;
        }
    }

    private void writeRemaining() throws IOException {
        while (buf.hasRemaining()) {
            int n = out.write(buf);
            if (n <= 0) {
                throw new IOException("failed to write to " + path + " at byte " + position);
            }
        }
    }

    private void checkCounts(long tensorCount, long kvCount) throws IOException {
        if (tensorCount < 0 || tensorCount > Integer.MAX_VALUE - 8L) {
            throw new IOException("model has an invalid tensor count " + tensorCount);
        }
        if (kvCount < 0 || kvCount > Integer.MAX_VALUE - 8L) {
            throw new IOException("model has an invalid key/value count " + kvCount);
        }
    }

    private byte[] utf8(String s, String owner) throws IOException {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        if (bytes.length > MAX_STRING_LENGTH) {
            throw new IOException(owner + " is " + bytes.length + " bytes long, more than the maximum of "
                    + MAX_STRING_LENGTH);
        }
        return bytes;
    }

    /** Rejects a name at or below the UTF-8 bytes of {@link GGUFTensorInfo#MAX_NAME_BYTES} (63 bytes). */
    private byte[] checkName(byte[] bytes, String name) throws IOException {
        if (bytes.length >= GGUFTensorInfo.MAX_NAME_BYTES) {
            throw new IOException("'" + name + "' is " + bytes.length + " bytes long, more than the "
                    + (GGUFTensorInfo.MAX_NAME_BYTES - 1) + " allowed for a GGUF name");
        }
        return bytes;
    }
}
