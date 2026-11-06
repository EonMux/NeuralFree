package com.eonmux.neuralfree.io.safetensors;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.EOFException;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The validated header of a safetensors file: {@code [u64 LE n][n bytes of UTF-8 JSON][data section]}, where the JSON
 * object maps each tensor name to {@code {"dtype", "shape", "data_offsets": [begin, end]}} (offsets relative to the
 * data section at {@code 8 + n}) and the optional {@code "__metadata__"} entry holds string metadata. Immutable.
 *
 * <p>{@link #read} and {@link #parse} enforce, in order: the length field ({@code 2 <= n <= 100,000,000} and {@code 8 + n <= size});
 * strict UTF-8 starting with {@code '{'} after whitespace; JSON under {@link Json.Limits#STRICT} (duplicate keys
 * rejected); string-valued metadata; exactly the three tensor fields; shapes of at most {@value #MAX_RANK} integer
 * dimensions in {@code [0, 2^53)} whose element count (checked multiplication) is a whole number of bytes; offsets
 * {@code 0 <= begin <= end} spanning exactly the tensor's bytes; tensors tiling {@code [0, total)} without gaps or
 * overlaps (zero-length tensors allowed) and {@code 8 + n + total == size}, so no trailing bytes (safetensors
 * tensor.rs {@code read_metadata}, {@code validate}). Every violation is an {@link IOException} whose message starts
 * with {@code "Invalid safetensors file '<path>': "} and names the tensor, key or offset.</p>
 */
public final class SafetensorsHeader {

    /** Byte length of the header length field. */
    public static final int LENGTH_FIELD_BYTES = 8;
    /** Smallest header accepted ({@code {}}). */
    public static final long MIN_HEADER_BYTES = 2;
    /** Largest header accepted (safetensors {@code MAX_HEADER_SIZE}). */
    public static final long MAX_HEADER_BYTES = 100_000_000L;
    /** Highest tensor rank accepted. */
    public static final int MAX_RANK = 8;
    /** Exclusive upper bound of a dimension: integers above it are not exact in every JSON implementation. */
    public static final long DIMENSION_LIMIT = 1L << 53;
    /** The reserved header key of the string metadata. */
    public static final String METADATA_KEY = "__metadata__";

    /** Tensor-name order of the reference implementation: Unicode code point (UTF-8 byte) order. */
    public static final Comparator<String> NAME_ORDER = SafetensorsHeader::compareCodePoints;

    private static final String DTYPE = "dtype";
    private static final String SHAPE = "shape";
    private static final String DATA_OFFSETS = "data_offsets";

    private final String path;
    private final long headerLength;
    private final long dataLength;
    private final Map<String, String> metadata;
    private final List<StTensor> tensors;
    private final Map<String, StTensor> byName;

    private SafetensorsHeader(String path, long headerLength, long dataLength, Map<String, String> metadata,
                              List<StTensor> tensors) {
        this.path = path;
        this.headerLength = headerLength;
        this.dataLength = dataLength;
        this.metadata = Collections.unmodifiableMap(metadata);
        List<StTensor> sorted = new ArrayList<>(tensors);
        sorted.sort((a, b) -> NAME_ORDER.compare(a.name(), b.name()));
        this.tensors = Collections.unmodifiableList(sorted);
        Map<String, StTensor> m = new HashMap<>(sorted.size() * 2);
        for (StTensor t : sorted) {
            m.put(t.name(), t);
        }
        this.byName = m;
    }

    /**
     * Reads and validates the header of the safetensors file open on {@code ch}.
     *
     * @param ch   a channel open for reading on the file; its position is not used or changed
     * @param path the file's path, for messages
     * @throws IOException when the file cannot be read or violates a rule of the class description
     */
    public static SafetensorsHeader read(FileChannel ch, String path) throws IOException {
        Objects.requireNonNull(ch, "ch");
        Objects.requireNonNull(path, "path");
        long size = ch.size();
        long n = readLength(ch, path, size);
        byte[] bytes = new byte[(int) n];
        readFully(ch, ByteBuffer.wrap(bytes), LENGTH_FIELD_BYTES, path);
        return fromHeaderBytes(bytes, n, size, path);
    }

    /**
     * Parses and validates a header that was fetched apart from its data section, as a hub client that reads only
     * the beginning of a remote shard does: {@code headerPrefix} is the {@value #LENGTH_FIELD_BYTES}-byte
     * little-endian length field followed by exactly the {@code n} header bytes it announces, and {@code fileSize}
     * is the size of the whole file the header belongs to. Every rule of {@link #read} applies, including the one
     * that the data section ends exactly at {@code fileSize}.
     *
     * @throws IOException when the prefix violates a rule of the class description
     */
    public static SafetensorsHeader parse(byte[] headerPrefix, long fileSize, String path) throws IOException {
        Objects.requireNonNull(headerPrefix, "headerPrefix");
        Objects.requireNonNull(path, "path");
        if (fileSize < 0) {
            throw new IllegalArgumentException("the size of '" + path + "' must be known, got " + fileSize);
        }
        if (headerPrefix.length < LENGTH_FIELD_BYTES) {
            throw invalid(path, "the fetched header has " + headerPrefix.length + " bytes, fewer than the "
                    + LENGTH_FIELD_BYTES + "-byte header length field");
        }
        long n = ByteBuffer.wrap(headerPrefix, 0, LENGTH_FIELD_BYTES).order(ByteOrder.LITTLE_ENDIAN).getLong(0);
        validateLength(n, fileSize, path);
        if (n != headerPrefix.length - LENGTH_FIELD_BYTES) {
            throw invalid(path, "the fetched header holds " + (headerPrefix.length - LENGTH_FIELD_BYTES)
                    + " bytes after its length field, but the field announces " + n);
        }
        byte[] bytes = Arrays.copyOfRange(headerPrefix, LENGTH_FIELD_BYTES, headerPrefix.length);
        return fromHeaderBytes(bytes, n, fileSize, path);
    }

    /** Validates the JSON bytes of a header of {@code n} bytes and returns the header. */
    private static SafetensorsHeader fromHeaderBytes(byte[] bytes, long n, long size, String path)
            throws IOException {
        Map<?, ?> root = parseJson(bytes, path);
        Map<String, String> metadata = new LinkedHashMap<>();
        List<StTensor> tensors = new ArrayList<>(root.size());
        for (Map.Entry<?, ?> e : root.entrySet()) {
            String name = (String) e.getKey();
            if (METADATA_KEY.equals(name)) {
                readMetadata(e.getValue(), path, metadata);
            } else {
                tensors.add(readTensor(name, e.getValue(), path));
            }
        }
        long total = checkTiling(tensors, path);
        long end = LENGTH_FIELD_BYTES + n;
        if (total > size - end) {
            throw invalid(path, "the data section of " + total + " bytes starting at byte " + end
                    + " runs past the end of the file (" + size + " bytes)");
        }
        end += total;
        if (end < size) {
            throw invalid(path, (size - end) + " trailing bytes follow the data section, which ends at byte " + end
                    + " of " + size);
        }
        return new SafetensorsHeader(path, n, total, metadata, tensors);
    }

    // ------------------------------------------------------------------ steps

    private static long readLength(FileChannel ch, String path, long size) throws IOException {
        if (size < LENGTH_FIELD_BYTES) {
            throw invalid(path, "the file has " + size + " bytes, fewer than the " + LENGTH_FIELD_BYTES
                    + "-byte header length field");
        }
        ByteBuffer len = ByteBuffer.allocate(LENGTH_FIELD_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(ch, len, 0, path);
        long n = len.getLong(0);
        validateLength(n, size, path);
        return n;
    }

    /** The length-field rules: {@code 2 <= n <= 100,000,000} and {@code 8 + n <= size}. */
    private static void validateLength(long n, long size, String path) throws IOException {
        if (n < 0 || n > MAX_HEADER_BYTES) {
            throw invalid(path, "header length " + Long.toUnsignedString(n) + " exceeds the limit of "
                    + MAX_HEADER_BYTES + " bytes");
        }
        if (n < MIN_HEADER_BYTES) {
            throw invalid(path, "header length " + n + " is below the minimum of " + MIN_HEADER_BYTES + " bytes");
        }
        if (n > size - LENGTH_FIELD_BYTES) {
            throw invalid(path, "header length " + n + " exceeds the file size (" + size + " bytes)");
        }
    }

    private static Map<?, ?> parseJson(byte[] bytes, String path) throws IOException {
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        ByteBuffer in = ByteBuffer.wrap(bytes);
        String text;
        try {
            text = decoder.decode(in).toString();
        } catch (CharacterCodingException e) {
            throw invalid(path, "the header is not valid UTF-8 (near header byte " + in.position() + ")", e);
        }
        int first = 0;
        while (first < bytes.length && isJsonSpace(bytes[first])) {
            first++;
        }
        if (first == bytes.length || bytes[first] != '{') {
            throw invalid(path, "the header does not start with '{'");
        }
        Object root;
        try {
            root = Json.parse(text, Json.Limits.STRICT);
        } catch (JsonException e) {
            throw invalid(path, "the header is not valid JSON: " + e.getMessage(), e);
        }
        if (!(root instanceof Map)) {
            throw invalid(path, "the header is not a JSON object");
        }
        return (Map<?, ?>) root;
    }

    private static void readMetadata(Object value, String path, Map<String, String> out) throws IOException {
        if (!(value instanceof Map)) {
            throw invalid(path, METADATA_KEY + " is " + typeOf(value) + ", not an object");
        }
        for (Map.Entry<?, ?> m : ((Map<?, ?>) value).entrySet()) {
            if (!(m.getValue() instanceof String)) {
                throw invalid(path, METADATA_KEY + " key '" + m.getKey() + "' is " + typeOf(m.getValue())
                        + ", not a string");
            }
            out.put((String) m.getKey(), (String) m.getValue());
        }
    }

    private static StTensor readTensor(String name, Object value, String path) throws IOException {
        if (!(value instanceof Map)) {
            throw invalid(path, "tensor '" + name + "' is " + typeOf(value) + ", not an object");
        }
        Map<?, ?> info = (Map<?, ?>) value;
        if (info.size() != 3 || !info.containsKey(DTYPE) || !info.containsKey(SHAPE)
                || !info.containsKey(DATA_OFFSETS)) {
            throw invalid(path, "tensor '" + name + "' has the fields " + info.keySet()
                    + "; expected exactly [dtype, shape, data_offsets]");
        }
        Object dtypeName = info.get(DTYPE);
        StDType dtype = dtypeName instanceof String ? StDType.fromName((String) dtypeName) : null;
        if (dtype == null) {
            throw invalid(path, "tensor '" + name + "' has the unknown dtype "
                    + (dtypeName instanceof String ? "'" + dtypeName + "'" : typeOf(dtypeName)));
        }
        long[] shape = readShape(name, info.get(SHAPE), path);
        long numel = 1;
        long bits;
        try {
            for (long d : shape) {
                numel = Math.multiplyExact(numel, d);
            }
            bits = Math.multiplyExact(numel, (long) dtype.bits());
        } catch (ArithmeticException e) {
            throw invalid(path, "tensor '" + name + "': the size of shape " + Arrays.toString(shape) + " of "
                    + dtype.headerName() + " overflows a 64-bit integer", e);
        }
        if (bits % 8 != 0) {
            throw invalid(path, "tensor '" + name + "': " + numel + " " + dtype.headerName() + " elements are "
                    + bits + " bits, not a whole number of bytes");
        }
        long[] offsets = readOffsets(name, info.get(DATA_OFFSETS), path);
        if (offsets[1] - offsets[0] != bits / 8) {
            throw invalid(path, "tensor '" + name + "': data_offsets [" + offsets[0] + ", " + offsets[1]
                    + "] span " + (offsets[1] - offsets[0]) + " bytes, but " + Arrays.toString(shape) + " of "
                    + dtype.headerName() + " needs " + (bits / 8));
        }
        return new StTensor(name, dtype, shape, offsets[0], offsets[1] - offsets[0]);
    }

    private static long[] readShape(String name, Object value, String path) throws IOException {
        if (!(value instanceof List)) {
            throw invalid(path, "tensor '" + name + "': shape is " + typeOf(value) + ", not an array");
        }
        List<?> dims = (List<?>) value;
        if (dims.size() > MAX_RANK) {
            throw invalid(path, "tensor '" + name + "': rank " + dims.size() + " exceeds the limit of " + MAX_RANK);
        }
        long[] shape = new long[dims.size()];
        for (int i = 0; i < shape.length; i++) {
            Object d = dims.get(i);
            if (!(d instanceof Long) || (Long) d < 0 || (Long) d >= DIMENSION_LIMIT) {
                throw invalid(path, "tensor '" + name + "': shape dimension " + i + " is " + describe(d)
                        + ", not an integer in [0, 2^53)");
            }
            shape[i] = (Long) d;
        }
        return shape;
    }

    private static long[] readOffsets(String name, Object value, String path) throws IOException {
        if (!(value instanceof List) || ((List<?>) value).size() != 2) {
            throw invalid(path, "tensor '" + name + "': data_offsets is " + describe(value)
                    + ", not an array of two integers");
        }
        List<?> pair = (List<?>) value;
        Object b = pair.get(0);
        Object e = pair.get(1);
        if (!(b instanceof Long) || !(e instanceof Long) || (Long) b < 0 || (Long) e < (Long) b) {
            throw invalid(path, "tensor '" + name + "': data_offsets " + describe(value)
                    + " is not a range [begin, end] with 0 <= begin <= end");
        }
        return new long[]{(Long) b, (Long) e};
    }

    /** Checks that the tensors tile {@code [0, total)} in offset order and returns {@code total}. */
    private static long checkTiling(List<StTensor> tensors, String path) throws IOException {
        List<StTensor> byOffset = new ArrayList<>(tensors);
        byOffset.sort(Comparator.comparingLong(StTensor::begin).thenComparingLong(StTensor::end));
        long end = 0;
        StTensor prev = null;
        for (StTensor t : byOffset) {
            if (t.begin() < end) {
                throw invalid(path, "tensor '" + t.name() + "' at data offsets [" + t.begin() + ", " + t.end()
                        + ") overlaps tensor '" + prev.name() + "', which ends at " + end);
            }
            if (t.begin() > end) {
                throw invalid(path, "tensor '" + t.name() + "' at data offsets [" + t.begin() + ", " + t.end()
                        + ") leaves the bytes [" + end + ", " + t.begin() + ") unused"
                        + (prev == null ? " at the start of the data section" : " after tensor '" + prev.name() + "'"));
            }
            end = t.end();
            prev = t;
        }
        return end;
    }

    // ------------------------------------------------------------------ accessors

    /** The file's path, as given to {@link #read}. */
    public String path() { return path; }

    /** Byte length {@code n} of the JSON header, padding included. */
    public long headerLength() { return headerLength; }

    /** File offset of the data section: {@code 8 + n}. */
    public long dataStart() { return LENGTH_FIELD_BYTES + headerLength; }

    /** Byte length of the data section: the end of the last tensor. */
    public long dataLength() { return dataLength; }

    /** The {@code __metadata__} entries in document order (unmodifiable; empty when absent). */
    public Map<String, String> metadata() { return metadata; }

    /** Every tensor, in {@link #NAME_ORDER} (the order of the reference {@code keys()}); unmodifiable. */
    public List<StTensor> tensors() { return tensors; }

    /** The tensor named {@code name}, or null when the file has none. */
    public StTensor tensor(String name) { return byName.get(name); }

    /** The number of tensors. */
    public int size() { return tensors.size(); }

    @Override
    public String toString() {
        return "SafetensorsHeader{" + path + ", " + tensors.size() + " tensors, data=[" + dataStart() + ", +"
                + dataLength + ")}";
    }

    // ------------------------------------------------------------------ helpers

    /** {@code IOException("Invalid safetensors file '<path>': <reason>")}. */
    static IOException invalid(String path, String reason) {
        return new IOException("Invalid safetensors file '" + path + "': " + reason);
    }

    /** As {@link #invalid(String, String)}, caused by {@code cause}. */
    static IOException invalid(String path, String reason, Throwable cause) {
        return new IOException("Invalid safetensors file '" + path + "': " + reason, cause);
    }

    /** Fills {@code dst} from {@code ch} at file position {@code pos}. */
    static void readFully(FileChannel ch, ByteBuffer dst, long pos, String path) throws IOException {
        long p = pos;
        while (dst.hasRemaining()) {
            int r = ch.read(dst, p);
            if (r < 0) {
                throw new EOFException("Invalid safetensors file '" + path + "': the file ends at byte " + p
                        + ", before the " + dst.remaining() + " further bytes being read");
            }
            p += r;
        }
    }

    private static boolean isJsonSpace(byte b) {
        return b == ' ' || b == '\t' || b == '\n' || b == '\r';
    }

    private static String typeOf(Object v) {
        if (v == null) return "null";
        if (v instanceof Map) return "an object";
        if (v instanceof List) return "an array";
        if (v instanceof String) return "a string";
        if (v instanceof Boolean) return "a boolean";
        if (v instanceof Long || v instanceof BigInteger) return "an integer";
        return "a number";
    }

    private static String describe(Object v) {
        if (v instanceof String) return "'" + v + "'";
        return v instanceof Map ? "an object" : String.valueOf(v);
    }

    private static int compareCodePoints(String a, String b) {
        int i = 0;
        int j = 0;
        while (i < a.length() && j < b.length()) {
            int ca = a.codePointAt(i);
            int cb = b.codePointAt(j);
            if (ca != cb) {
                return Integer.compare(ca, cb);
            }
            i += Character.charCount(ca);
            j += Character.charCount(cb);
        }
        return Integer.compare(a.length() - i, b.length() - j);
    }
}
