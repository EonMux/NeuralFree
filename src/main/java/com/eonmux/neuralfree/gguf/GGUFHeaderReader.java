package com.eonmux.neuralfree.gguf;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Little-endian reader for one GGUF header: magic, version, tensor and key/value counts, the key/value section and
 * the tensor infos. It applies the checks of llama.cpp ggml/src/gguf.cpp {@code gguf_reader} and
 * {@code gguf_init_from_file_impl}: the version (gguf.cpp:493-520), the entry counts (gguf.cpp:524-546), empty and
 * duplicate keys (gguf.cpp:572-580), nested arrays (gguf.cpp:606-611), tensor name length and rank
 * (gguf.cpp:657-680), and every string length and array count against its maximum and the bytes left in the file
 * before anything is allocated (gguf.cpp:280-296, 348-356).
 *
 * <p>Reads are positional channel reads through a private buffer, so the channel position and any file pointer
 * of the underlying file are left untouched. Not thread-safe.</p>
 */
final class GGUFHeaderReader {

    /** Newest GGUF version this reader understands ({@code GGUF_VERSION}, ggml/include/gguf.h:42). */
    static final int GGUF_VERSION = 3;

    /** Longest string in bytes ({@code GGUF_MAX_STRING_LENGTH}, gguf.cpp:20). */
    static final long MAX_STRING_LENGTH = 1L << 30;

    /** Largest array element count ({@code GGUF_MAX_ARRAY_ELEMENTS}, gguf.cpp:21). */
    static final long MAX_ARRAY_ELEMENTS = 1L << 30;

    /** Maximum tensor rank ({@code GGML_MAX_DIMS}). */
    static final int MAX_DIMS = 4;

    /** Tensor names must be shorter than this many bytes ({@code GGML_MAX_NAME}). */
    static final int MAX_NAME = 64;

    /** Largest tensor or key/value count one Java collection indexes. */
    private static final long MAX_ENTRIES = Integer.MAX_VALUE - 8L;

    /** Smallest key/value pair: key length, a one-byte key, the type id and a one-byte value. */
    private static final long MIN_KV_BYTES = 8 + 1 + 4 + 1;

    /** Smallest tensor info: name length, rank, type id and data offset. */
    private static final long MIN_TENSOR_INFO_BYTES = 8 + 4 + 4 + 8;

    private static final int BUFFER_BYTES = 1 << 16;

    /** Capacity an array list starts with; larger arrays grow as their elements are read. */
    private static final long INITIAL_ARRAY_CAPACITY = 1 << 16;

    /** Version and entry counts of a header, validated as llama.cpp does. */
    static final class Header {
        final int version;
        final long tensorCount;
        final long kvCount;

        private Header(int version, long tensorCount, long kvCount) {
            this.version = version;
            this.tensorCount = tensorCount;
            this.kvCount = kvCount;
        }
    }

    private final String path;
    private final FileChannel channel;
    private final long fileLength;
    private final ByteBuffer buf;
    private long position;

    /** A reader positioned at the start of the file behind {@code channel}; {@code path} names it in messages. */
    GGUFHeaderReader(String path, FileChannel channel) throws IOException {
        this.path = path;
        this.channel = channel;
        this.fileLength = channel.size();
        this.buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.buf.flip();
    }

    /** File offset of the next byte to read; after the tensor infos, the end of the header. */
    long position() {
        return position;
    }

    // ------------------------------------------------------------------ header sections

    /** Reads and validates magic, version and the two entry counts. */
    Header readHeader() throws IOException {
        byte[] magic = new byte[4];
        readFully(magic);
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
            throw format("bad magic '" + new String(magic, StandardCharsets.ISO_8859_1) + "', expected '"
                    + GGUFFileParser.MAGIC + "'");
        }
        int version = u32();
        validateVersion(version);
        long nTensors = u64();
        long tensorLimit = Math.min(MAX_ENTRIES, remaining() / MIN_TENSOR_INFO_BYTES);
        if (nTensors < 0 || nTensors > tensorLimit) {
            throw format("number of tensors is " + Long.toUnsignedString(nTensors) + " but must be in [0, "
                    + tensorLimit + "] for a file of " + fileLength + " bytes");
        }
        long nKv = u64();
        long kvLimit = Math.min(MAX_ENTRIES,
                Math.max(0L, remaining() - nTensors * MIN_TENSOR_INFO_BYTES) / MIN_KV_BYTES);
        if (nKv < 0 || nKv > kvLimit) {
            throw format("number of key/value pairs is " + Long.toUnsignedString(nKv) + " but must be in [0, "
                    + kvLimit + "] for a file of " + fileLength + " bytes and " + nTensors + " tensors");
        }
        return new Header(version, nTensors, nKv);
    }

    /**
     * Reads key/value pair {@code index} into {@code values}, recording its type in {@code types} and, for an
     * array, the element type in {@code elementTypes} (which may be null).
     */
    void readKVPair(long index, Map<String, Object> values, Map<String, GGUFValueType> types,
                    Map<String, GGUFValueType> elementTypes) throws IOException {
        String key = readString("key " + index, -1L);
        if (key.isEmpty()) {
            throw format("key " + index + " is empty");
        }
        if (values.containsKey(key)) {
            throw format("duplicate key '" + key + "' for key/value pair " + index);
        }
        GGUFValueType type = valueType(key);
        if (type == GGUFValueType.ARRAY) {
            GGUFValueType elementType = valueType(key);
            if (elementType == GGUFValueType.ARRAY) {
                throw format("key '" + key + "' has invalid GGUF type: an array of arrays");
            }
            values.put(key, readArrayBody(key, elementType, u64()));
            if (elementTypes != null) {
                elementTypes.put(key, elementType);
            }
        } else {
            values.put(key, readScalar(type, "key '" + key + "'", -1L));
        }
        types.put(key, type);
    }

    /** Reads tensor info {@code index}: name, rank, extents, ggml type and data offset. */
    GGUFTensorInfo readTensorInfo(long index) throws IOException {
        byte[] nameBytes = readStringBytes("name of tensor " + index, -1L);
        if (nameBytes.length >= MAX_NAME) {
            throw format("name of tensor " + index + " is " + nameBytes.length + " bytes long, more than the "
                    + (MAX_NAME - 1) + " allowed");
        }
        String name = new String(nameBytes, StandardCharsets.UTF_8);
        int nDims = u32();
        if (Integer.compareUnsigned(nDims, MAX_DIMS) > 0) {
            throw format("tensor '" + name + "' has " + Integer.toUnsignedString(nDims)
                    + " dimensions, more than the maximum of " + MAX_DIMS);
        }
        long[] dims = new long[nDims];
        for (int i = 0; i < nDims; i++) {
            dims[i] = u64();
        }
        GGMLType type = GGMLType.fromId(u32());
        long offset = u64();
        return new GGUFTensorInfo(name, dims, type, offset);
    }

    // ------------------------------------------------------------------ values

    private GGUFValueType valueType(String key) throws IOException {
        int id = u32();
        try {
            return GGUFValueType.fromId(id);
        } catch (IllegalArgumentException e) {
            throw format("key '" + key + "' has invalid GGUF type " + Integer.toUnsignedString(id));
        }
    }

    /** One scalar value in the decoded representation documented on {@link GGUFValueType}. */
    private Object readScalar(GGUFValueType type, String owner, long element) throws IOException {
        switch (type) {
            case UINT8:   return u8();
            case INT8:    return (int) (byte) u8();
            case UINT16:  return u16();
            case INT16:   return (int) (short) u16();
            case UINT32:
            case INT32:   return u32();
            case FLOAT32: return Float.intBitsToFloat(u32());
            case BOOL:    return u8() != 0;
            case STRING:  return readString(owner, element);
            case UINT64:
            case INT64:   return u64();
            case FLOAT64: return Double.longBitsToDouble(u64());
            default:
                throw format(describe(owner, element) + " has invalid GGUF type " + type);
        }
    }

    private List<Object> readArrayBody(String key, GGUFValueType elementType, long count) throws IOException {
        String owner = "array key '" + key + "'";
        if (count < 0 || count > MAX_ARRAY_ELEMENTS) {
            throw format(owner + " has " + Long.toUnsignedString(count) + " elements, more than the maximum of "
                    + MAX_ARRAY_ELEMENTS);
        }
        long minBytes = count * minElementBytes(elementType);
        if (minBytes > remaining()) {
            throw format(owner + " has " + count + " " + elementType + " elements needing at least " + minBytes
                    + " bytes, but only " + remaining() + " bytes are left in the file");
        }
        List<Object> arr = new ArrayList<>((int) Math.min(count, INITIAL_ARRAY_CAPACITY));
        for (long k = 0; k < count; k++) {
            arr.add(readScalar(elementType, owner, k));
        }
        return arr;
    }

    /** Bytes one element of {@code type} occupies at least: its size, or the length prefix of a string. */
    private static int minElementBytes(GGUFValueType type) {
        switch (type) {
            case UINT8: case INT8: case BOOL:
                return 1;
            case UINT16: case INT16:
                return 2;
            case UINT32: case INT32: case FLOAT32:
                return 4;
            default:
                return 8;
        }
    }

    private String readString(String owner, long element) throws IOException {
        return new String(readStringBytes(owner, element), StandardCharsets.UTF_8);
    }

    private byte[] readStringBytes(String owner, long element) throws IOException {
        long len = u64();
        if (len < 0 || len > MAX_STRING_LENGTH) {
            throw format("string length " + Long.toUnsignedString(len) + " of " + describe(owner, element)
                    + " exceeds the maximum of " + MAX_STRING_LENGTH);
        }
        if (len > remaining()) {
            throw format("string length " + len + " of " + describe(owner, element) + " exceeds the "
                    + remaining() + " bytes left in the file");
        }
        byte[] bytes = new byte[(int) len];
        readFully(bytes);
        return bytes;
    }

    private static String describe(String owner, long element) {
        return element < 0 ? owner : "element " + element + " of " + owner;
    }

    // ------------------------------------------------------------------ primitives

    private int u8() throws IOException {
        ensure(1);
        position += 1;
        return buf.get() & 0xFF;
    }

    private int u16() throws IOException {
        ensure(2);
        position += 2;
        return buf.getShort() & 0xFFFF;
    }

    private int u32() throws IOException {
        ensure(4);
        position += 4;
        return buf.getInt();
    }

    private long u64() throws IOException {
        ensure(8);
        position += 8;
        return buf.getLong();
    }

    private void readFully(byte[] dst) throws IOException {
        int off = 0;
        while (off < dst.length) {
            ensure(Math.min(BUFFER_BYTES, dst.length - off));
            int n = Math.min(buf.remaining(), dst.length - off);
            buf.get(dst, off, n);
            off += n;
            position += n;
        }
    }

    private long remaining() {
        return fileLength - position;
    }

    /** Makes at least {@code n} (at most the buffer size) bytes available in the buffer. */
    private void ensure(int n) throws IOException {
        if (buf.remaining() >= n) {
            return;
        }
        if (n > remaining()) {
            throw new EOFException("Invalid GGUF file '" + path + "': the header ends at byte " + fileLength
                    + ", before the " + n + " bytes needed at byte " + position);
        }
        buf.compact();
        long readAt = position + buf.position();
        while (buf.position() < n) {
            int r = channel.read(buf, readAt);
            if (r < 0) {
                buf.flip();
                throw new EOFException("Invalid GGUF file '" + path + "': the file ended at byte " + readAt
                        + " while its header was read");
            }
            readAt += r;
        }
        buf.flip();
    }

    private void validateVersion(int version) throws IOException {
        if (version == 0) {
            throw format("bad GGUF version: 0");
        }
        if ((version & 0xFFFF) == 0) {
            throw format("GGUF version " + Integer.toUnsignedString(version)
                    + " is extremely large; is there a mismatch between the host and model endianness?");
        }
        if (version == 1) {
            throw format("GGUFv1 is no longer supported; convert the model with an up-to-date llama.cpp");
        }
        if (Integer.compareUnsigned(version, GGUF_VERSION) > 0) {
            throw format("this GGUF file is version " + Integer.toUnsignedString(version)
                    + " but only versions up to " + GGUF_VERSION + " are supported");
        }
    }

    private IOException format(String detail) {
        return new IOException("Invalid GGUF file '" + path + "': " + detail);
    }
}
