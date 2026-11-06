package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reader of CoreWeave tensorizer's serialized format (the {@code tensorizer} library's
 * {@code TensorSerializer.write_module}/{@code write_state_dict}).
 *
 * <p>Layout, little-endian throughout:</p>
 * <ol>
 *   <li>the magic {@code |TZR|} and a {@code uint32} data version (1 to 5);</li>
 *   <li>a file header: 32 feature-flag bytes, {@code uint64} tensor-data size, 8 unused bytes and {@code uint64}
 *       tensor count. From version 4 the feature flags select encryption and versioned metadata;</li>
 *   <li>the metadata index: {@code uint64} byte length, then one entry per tensor — the optional {@code uint32}
 *       metadata version, the name, the tensor type, the dtype string, the shape and the header, data and data-length
 *       offsets;</li>
 *   <li>one per-tensor header and its data, repeated in tensor count order. A header repeats the name, type, dtype
 *       and shape, holds the hashes and the optional crypt info, and ends with the {@code int64} data length.</li>
 * </ol>
 *
 * <p>Every name, length and offset is validated against the file size before it is used. A meta tensor (no data)
 * other than vLLM's marker is refused; a dtype without a safetensors equivalent is refused by name. Encrypted tensors
 * are decrypted with the 32-byte key from {@code NEURALFREE_TENSORIZER_KEY} (64 hex digits), or from the file named
 * by {@code NEURALFREE_TENSORIZER_KEY_FILE}; without a key an encrypted file is refused by name. The per-chunk
 * XSalsa20-Poly1305 tag is verified before any plaintext is served.</p>
 */
final class TensorizerFile implements AutoCloseable {

    /** The environment variable holding a 32-byte secretbox key as 64 hex digits. */
    static final String KEY_ENV = "NEURALFREE_TENSORIZER_KEY";
    /** The environment variable holding a file with the 32-byte key, as 64 hex digits or raw bytes. */
    static final String KEY_FILE_ENV = "NEURALFREE_TENSORIZER_KEY_FILE";

    private static final byte[] MAGIC = {'|', 'T', 'Z', 'R', '|'};
    private static final int MAGIC_LEN = 5;
    private static final int FILE_HEADER_LEN = 56;
    private static final int MAX_VERSION = 5;
    private static final int VERSIONED_HEADERS = 2;
    private static final int ENCRYPTED_FLAG = 1;
    private static final int META_VERSION = 4;
    private static final int LONG_VERSION = 5;
    private static final int ENCRYPTION_VERSION = 3;
    private static final int MAX_HEADER_BYTES = 1 << 24;
    private static final int MAX_METADATA_BYTES = 1 << 30;
    private static final int MAX_DIMENSIONS = 64;

    private final Path path;
    private final FileChannel channel;
    private final List<StoredTensor> tensors;

    private TensorizerFile(Path path, FileChannel channel, List<StoredTensor> tensors) {
        this.path = path;
        this.channel = channel;
        this.tensors = Collections.unmodifiableList(tensors);
    }

    /** The tensors of the file, in metadata order; a marker tensor is not listed. */
    List<StoredTensor> tensors() {
        return tensors;
    }

    Path path() {
        return path;
    }

    /**
     * Reads the tensorizer file at {@code path}.
     *
     * @throws IOException when the file is not a tensorizer file, its layout is invalid, it is encrypted and no key
     *                     is configured, or a tensor's dtype or size is unsupported
     */
    static TensorizerFile open(Path path) throws IOException {
        return open(path, null);
    }

    /**
     * Reads the tensorizer file at {@code path}, decrypting with {@code key} when it is encrypted.
     *
     * @param key the 32-byte key, or null to take it from {@link #KEY_ENV}/{@link #KEY_FILE_ENV}
     */
    static TensorizerFile open(Path path, byte[] key) throws IOException {
        Objects.requireNonNull(path, "path");
        if (key != null && key.length != TensorizerCrypto.KEY_BYTES) {
            throw new IllegalArgumentException("a tensorizer key is " + TensorizerCrypto.KEY_BYTES + " bytes, got "
                    + key.length);
        }
        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
        try {
            return parse(path, channel, key == null ? null : key.clone());
        } catch (IOException | RuntimeException e) {
            try {
                channel.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    private static TensorizerFile parse(Path path, FileChannel ch, byte[] configuredKey) throws IOException {
        long size = ch.size();
        byte[] magic = read(ch, 0, MAGIC_LEN, size, path);
        if (!Arrays.equals(magic, MAGIC)) {
            throw invalid(path, "it does not start with the tensorizer magic |TZR|");
        }
        long pos = MAGIC_LEN;
        int version = (int) uint32(read(ch, pos, 4, size, path));
        pos += 4;
        if (version < 1 || version > MAX_VERSION) {
            throw invalid(path, "it uses tensorizer data version " + version + ", not one of 1.." + MAX_VERSION);
        }
        byte[] fileHeader = read(ch, pos, FILE_HEADER_LEN, size, path);
        pos += FILE_HEADER_LEN;
        long flags = uint32(Arrays.copyOfRange(fileHeader, 0, 4));
        long tensorSize = uint64(fileHeader, 32);
        long tensorCount = uint64(fileHeader, 48);
        boolean encrypted = (flags & ENCRYPTED_FLAG) != 0 || version == ENCRYPTION_VERSION;
        boolean versioned = (flags & VERSIONED_HEADERS) != 0;
        if (versioned && version < LONG_VERSION) {
            throw invalid(path, "the versioned-metadata feature flag is set on a version " + version + " file");
        }
        if (tensorCount < 0 || tensorCount > Integer.MAX_VALUE) {
            throw invalid(path, "it declares " + tensorCount + " tensors");
        }
        byte[] key = encrypted ? (configuredKey != null ? configuredKey : key(path)) : null;
        long metadataLen = uint64(read(ch, pos, 8, size, path), 0);
        pos += 8;
        if (metadataLen < 0 || metadataLen > MAX_METADATA_BYTES || metadataLen > size - pos) {
            throw invalid(path, "its metadata index length " + metadataLen + " does not fit the file");
        }
        byte[] metadata = read(ch, pos, (int) metadataLen, size, path);
        pos += metadataLen;
        LinkedHashMap<String, Entry> entries = parseMetadata(path, metadata, (int) tensorCount, versioned);
        long tensorsBegin = pos;

        Map<String, RowSource> sources = new LinkedHashMap<>();
        for (int i = 0; i < tensorCount; i++) {
            pos = readTensor(path, ch, size, pos, tensorsBegin, version, encrypted || version == ENCRYPTION_VERSION,
                    key, entries, sources);
        }
        if (pos > size) {
            throw invalid(path, "its tensor data ends at " + pos + ", past the file's " + size + " bytes");
        }
        List<StoredTensor> out = new ArrayList<>(entries.size());
        for (Entry e : entries.values()) {
            if (e.marker) {
                continue;
            }
            RowSource source = sources.get(e.name);
            if (source == null) {
                throw invalid(path, "the index lists tensor '" + e.name + "' but no tensor header carries it");
            }
            out.add(StoredTensor.of(e.name, source));
        }
        return new TensorizerFile(path, ch, out);
    }

    // ------------------------------------------------------------------ metadata

    /** One metadata-index entry. */
    private static final class Entry {
        final String name;
        final String dtype;
        final long[] shape;
        final int metadataVersion;
        final long headerOffset;
        final long dataOffset;
        final long dataLength;
        final boolean marker;

        Entry(String name, String dtype, long[] shape, int metadataVersion, long headerOffset, long dataOffset,
              long dataLength) {
            this.name = name;
            this.dtype = dtype;
            this.shape = shape;
            this.metadataVersion = metadataVersion;
            this.headerOffset = headerOffset;
            this.dataOffset = dataOffset;
            this.dataLength = dataLength;
            this.marker = name.equals(VllmStateDict.TENSORIZED_MARKER) || name.endsWith(VllmStateDict.TENSORIZED_MARKER);
        }
    }

    private static LinkedHashMap<String, Entry> parseMetadata(Path path, byte[] buffer, int count, boolean versioned)
            throws IOException {
        LinkedHashMap<String, Entry> out = new LinkedHashMap<>();
        int offset = 0;
        for (int i = 0; i < count; i++) {
            int metadataVersion = 1;
            if (versioned) {
                if (offset > buffer.length - 4) {
                    throw invalid(path, "the metadata index ends inside tensor " + i + "'s version");
                }
                metadataVersion = (int) uint32(Arrays.copyOfRange(buffer, offset, offset + 4));
                offset += 4;
                if (metadataVersion < 1 || metadataVersion > MAX_VERSION) {
                    throw invalid(path, "tensor " + i + " uses metadata version " + metadataVersion);
                }
            }
            boolean longShape = metadataVersion >= LONG_VERSION;
            int nameAt = offset;
            String name = readName(path, buffer, offset, "tensor " + i);
            offset = nameEnd(buffer, nameAt);
            int type = byteAt(path, buffer, offset, "tensor " + i + "'s type");
            offset += 1;
            if (type < 0 || type > 2) {
                throw invalid(path, "tensor " + i + " has type " + type + ", not PARAM, BUFFER or STATE_DICT");
            }
            String dtype = readDtype(path, buffer, offset);
            offset += 1 + dtype.length();
            long[] shape = readShape(path, buffer, offset, longShape);
            offset += 1 + shape.length * (longShape ? 8 : 4);
            if (offset > buffer.length - 24) {
                throw invalid(path, "the metadata index ends inside tensor '" + name + "'s offsets");
            }
            long headerOffset = uint64(buffer, offset);
            long dataOffset = uint64(buffer, offset + 8);
            long dataLength = uint64(buffer, offset + 16);
            offset += 24;
            Entry e = new Entry(name, dtype, shape, metadataVersion, headerOffset, dataOffset, dataLength);
            if (out.put(name, e) != null) {
                throw invalid(path, "the metadata index lists tensor '" + name + "' twice");
            }
        }
        // the writer reserves a fixed metadata area and zero-fills the unused tail
        for (int i = offset; i < buffer.length; i++) {
            if (buffer[i] != 0) {
                throw invalid(path, "the metadata index has non-zero bytes past its last tensor");
            }
        }
        return out;
    }

    /** The end offset of a {@code uint16}-length-prefixed name that starts at {@code start}. */
    private static int nameEnd(byte[] buffer, int start) throws IOException {
        int len = uint16(buffer, start);
        return start + 2 + len;
    }

    private static String readName(Path path, byte[] buffer, int offset, String what) throws IOException {
        if (offset > buffer.length - 2) {
            throw invalid(path, what + " has no name length");
        }
        int len = uint16(buffer, offset);
        if (len <= 0 || offset + 2 + len > buffer.length) {
            throw invalid(path, what + " has an invalid name length " + len);
        }
        byte[] raw = Arrays.copyOfRange(buffer, offset + 2, offset + 2 + len);
        if (raw[0] == 0x1e) {
            throw invalid(path, what + " uses a structured (JSON-seq) tensor path, not a plain name");
        }
        String name = new String(raw, StandardCharsets.UTF_8);
        if (name.indexOf('\0') >= 0) {
            throw invalid(path, what + " has a NUL in its name");
        }
        return name;
    }

    private static String readDtype(Path path, byte[] buffer, int offset) throws IOException {
        if (offset > buffer.length - 1) {
            throw invalid(path, "a tensor has no dtype length");
        }
        int len = buffer[offset] & 0xff;
        if (len <= 0 || offset + 1 + len > buffer.length) {
            throw invalid(path, "a tensor has an invalid dtype length " + len);
        }
        return new String(buffer, offset + 1, len, StandardCharsets.UTF_8);
    }

    private static long[] readShape(Path path, byte[] buffer, int offset, boolean longShape) throws IOException {
        if (offset > buffer.length - 1) {
            throw invalid(path, "a tensor has no shape");
        }
        int rank = buffer[offset] & 0xff;
        int width = longShape ? 8 : 4;
        if (rank > MAX_DIMENSIONS || offset + 1 + rank * width > buffer.length) {
            throw invalid(path, "a tensor has an invalid rank " + rank);
        }
        long[] shape = new long[rank];
        for (int i = 0; i < rank; i++) {
            int at = offset + 1 + i * width;
            shape[i] = longShape ? uint64(buffer, at) : uint32(Arrays.copyOfRange(buffer, at, at + 4));
            if (shape[i] < 0) {
                throw invalid(path, "a tensor has a negative dimension " + shape[i]);
            }
        }
        return shape;
    }

    // ------------------------------------------------------------------ per-tensor headers

    /**
     * Reads one per-tensor header and records its source, returning the position after its data.
     */
    private static long readTensor(Path path, FileChannel ch, long size, long headerStart, long tensorsBegin,
                                   int version, boolean hasCryptInfo, byte[] key,
                                   Map<String, Entry> entries, Map<String, RowSource> sources) throws IOException {
        if (headerStart > size - 8) {
            throw invalid(path, "a tensor header starts at " + headerStart + ", past the file's " + size
                    + " bytes");
        }
        long headerLen = uint64(read(ch, headerStart, 8, size, path), 0);
        if (headerLen < 8 || headerLen > MAX_HEADER_BYTES || headerLen > size - headerStart) {
            throw invalid(path, "a tensor header length " + headerLen + " does not fit the file");
        }
        byte[] header = read(ch, headerStart, (int) headerLen, size, path);
        int offset = 8;
        offset += 2; // module index
        int type = byteAt(path, header, offset, "a tensor header's type");
        offset += 1;
        if (type < 0 || type > 2) {
            throw invalid(path, "a tensor header has type " + type);
        }
        String name = readName(path, header, offset, "a tensor header");
        offset = nameEnd(header, offset);
        String dtype = readDtype(path, header, offset);
        offset += 1 + dtype.length();
        Entry entry = entries.get(name);
        if (entry == null) {
            throw invalid(path, "the tensor header '" + name + "' is not in the metadata index");
        }
        boolean longShape = entry.metadataVersion >= LONG_VERSION;
        long[] shape = readShape(path, header, offset, longShape);
        offset += 1 + shape.length * (longShape ? 8 : 4);
        if (offset > header.length - 2) {
            throw invalid(path, "tensor '" + name + "' has no hash section");
        }
        int hashLen = uint16(header, offset);
        offset += 2;
        if (hashLen > header.length - offset) {
            throw invalid(path, "tensor '" + name + "' has a hash section of " + hashLen + " bytes");
        }
        offset += hashLen;
        Crypt crypt = null;
        if (hasCryptInfo) {
            if (offset > header.length - 8) {
                throw invalid(path, "tensor '" + name + "' has no crypt-info length");
            }
            long cryptLen = uint64(header, offset);
            offset += 8;
            if (cryptLen > header.length - offset) {
                throw invalid(path, "tensor '" + name + "' has a crypt-info block of " + cryptLen + " bytes");
            }
            crypt = parseCrypt(path, header, offset, (int) cryptLen, entry.dataLength);
            offset += (int) cryptLen;
        }
        if (offset != header.length - 8) {
            throw invalid(path, "tensor '" + name + "' has " + (header.length - 8 - offset)
                    + " bytes before its data length");
        }
        long dataLen = int64(header, header.length - 8);
        if (dataLen < 0 || dataLen != entry.dataLength) {
            throw invalid(path, "tensor '" + name + "' header declares " + dataLen + " data bytes, the index "
                    + entry.dataLength);
        }
        if (entry.headerOffset != headerStart) {
            throw invalid(path, "tensor '" + name + "' starts at " + headerStart + ", the index says "
                    + entry.headerOffset);
        }
        long dataStart = Math.addExact(headerStart, headerLen);
        if (entry.dataOffset != dataStart) {
            throw invalid(path, "tensor '" + name + "' data starts at " + dataStart + ", the index says "
                    + entry.dataOffset);
        }
        if (dataStart > size - dataLen) {
            throw invalid(path, "tensor '" + name + "' data [" + dataStart + ", +" + dataLen + ") is outside the"
                    + " file's " + size + " bytes");
        }
        if (!entry.marker) {
            StDType st = TensorizerDtype.parse(path, name, dtype);
            long expected = expectedBytes(path, name, st, shape);
            if (dataLen == 0 && expected > 0) {
                throw invalid(path, "tensor '" + name + "' is a meta tensor with no data");
            }
            if (dataLen != expected) {
                throw invalid(path, "tensor '" + name + "' holds " + dataLen + " data bytes, not the " + expected
                        + " of its " + st.headerName() + " " + Arrays.toString(shape));
            }
            if (crypt != null && key == null) {
                throw new IOException("Invalid tensorizer file '" + path + "': tensor '" + name + "' is encrypted;"
                        + " set " + KEY_ENV + " (64 hex digits) or " + KEY_FILE_ENV + " to read it");
            }
            sources.put(name, new TensorizerRowSource(name, ch, dataStart, dataLen, st, shape, crypt, key));
        }
        return Math.addExact(dataStart, dataLen);
    }

    private static long expectedBytes(Path path, String name, StDType dtype, long[] shape) throws IOException {
        long numel = 1;
        try {
            for (long d : shape) {
                numel = Math.multiplyExact(numel, d);
            }
            long bits = Math.multiplyExact(numel, (long) dtype.bits());
            if (bits % 8 != 0) {
                throw invalid(path, "tensor '" + name + "' is not a whole number of bytes");
            }
            return bits / 8;
        } catch (ArithmeticException e) {
            throw invalid(path, "tensor '" + name + "' overflows a 64-bit element count");
        }
    }

    // ------------------------------------------------------------------ crypt info

    /** How one tensor's data is encrypted. */
    abstract static class Crypt {
        abstract byte[] plaintext(FileChannel ch, long dataStart, long dataLength, byte[] key) throws IOException;
    }

    private static Crypt parseCrypt(Path path, byte[] buffer, int offset, int length, long dataLength)
            throws IOException {
        int end = offset + length;
        if (offset > end - 2) {
            throw invalid(path, "a crypt-info block is empty");
        }
        int chunks = uint16(buffer, offset);
        offset += 2;
        Crypt seq = null;
        Crypt chunked = null;
        for (int i = 0; i < chunks; i++) {
            if (offset > end - 8) {
                throw invalid(path, "a crypt-info chunk has no length");
            }
            long chunkLen = uint64(buffer, offset);
            offset += 8;
            if (chunkLen < 2 || chunkLen > end - offset) {
                throw invalid(path, "a crypt-info chunk has an invalid length " + chunkLen);
            }
            int chunkEnd = (int) (offset + chunkLen);
            int type = uint16(buffer, offset);
            offset += 2;
            if (type == 2) {
                chunked = parallelCrypt(path, buffer, offset, chunkEnd, dataLength);
            } else if (type == 3) {
                seq = sequentialCrypt(path, buffer, offset, chunkEnd);
            }
            offset = chunkEnd;
        }
        if (seq != null && chunked != null) {
            throw invalid(path, "a tensor carries two encryption methods");
        }
        if (seq == null && chunked == null) {
            if (dataLength == 0) {
                // a meta tensor carries an empty crypt info; it serves no data
                return null;
            }
            throw invalid(path, "a tensor is flagged encrypted but carries no known encryption method");
        }
        return seq != null ? seq : chunked;
    }

    private static Crypt sequentialCrypt(Path path, byte[] buffer, int offset, int end) throws IOException {
        if (end - offset != TensorizerCrypto.NONCE_BYTES + TensorizerCrypto.MAC_BYTES) {
            throw invalid(path, "a sequential crypt chunk has " + (end - offset) + " bytes");
        }
        byte[] nonce = Arrays.copyOfRange(buffer, offset, offset + TensorizerCrypto.NONCE_BYTES);
        byte[] mac = Arrays.copyOfRange(buffer, offset + TensorizerCrypto.NONCE_BYTES, end);
        return new SequentialCrypt(nonce, mac);
    }

    private static Crypt parallelCrypt(Path path, byte[] buffer, int offset, int end, long dataLength)
            throws IOException {
        int header = 8 + TensorizerCrypto.NONCE_BYTES + 8;
        if (end - offset < header) {
            throw invalid(path, "a parallel crypt chunk is too short");
        }
        long chunkSize = uint64(buffer, offset);
        byte[] nonce = Arrays.copyOfRange(buffer, offset + 8, offset + 8 + TensorizerCrypto.NONCE_BYTES);
        long macs = uint64(buffer, offset + 8 + TensorizerCrypto.NONCE_BYTES);
        offset += header;
        if (chunkSize < 1024 || chunkSize % 64 != 0) {
            throw invalid(path, "a parallel crypt chunk has chunk size " + chunkSize);
        }
        if (macs <= 0 || macs > Integer.MAX_VALUE || macs * TensorizerCrypto.MAC_BYTES != end - offset) {
            throw invalid(path, "a parallel crypt chunk declares " + macs + " MACs for " + (end - offset)
                    + " bytes");
        }
        long chunks = dataLength / chunkSize + (dataLength % chunkSize == 0 ? 0 : 1);
        if (chunks != macs) {
            throw invalid(path, "a tensor's data needs " + chunks + " chunks but its crypt info holds " + macs
                    + " MACs");
        }
        List<byte[]> list = new ArrayList<>((int) macs);
        for (int i = 0; i < macs; i++) {
            int at = offset + i * TensorizerCrypto.MAC_BYTES;
            list.add(Arrays.copyOfRange(buffer, at, at + TensorizerCrypto.MAC_BYTES));
        }
        return new ChunkedCrypt(chunkSize, nonce, list);
    }

    /** One XSalsa20-Poly1305 secretbox over the whole tensor. */
    private static final class SequentialCrypt extends Crypt {
        private final byte[] nonce;
        private final byte[] mac;
        private volatile byte[] plaintext;

        SequentialCrypt(byte[] nonce, byte[] mac) {
            this.nonce = nonce;
            this.mac = mac;
        }

        @Override
        byte[] plaintext(FileChannel ch, long dataStart, long dataLength, byte[] key) throws IOException {
            byte[] p = plaintext;
            if (p == null) {
                synchronized (this) {
                    p = plaintext;
                    if (p == null) {
                        if (dataLength > Integer.MAX_VALUE) {
                            throw new IOException("an encrypted tensor of " + dataLength + " bytes is too large to"
                                    + " decrypt as one chunk");
                        }
                        p = read(ch, dataStart, (int) dataLength, Long.MAX_VALUE, null);
                        TensorizerCrypto.openDetached(key, nonce, mac, p);
                        plaintext = p;
                    }
                }
            }
            return p;
        }
    }

    /** Per-chunk XSalsa20-Poly1305 secretboxes. */
    private static final class ChunkedCrypt extends Crypt {
        private final long chunkSize;
        private final byte[] initialNonce;
        private final List<byte[]> macs;

        ChunkedCrypt(long chunkSize, byte[] initialNonce, List<byte[]> macs) {
            this.chunkSize = chunkSize;
            this.initialNonce = initialNonce;
            this.macs = macs;
        }

        @Override
        byte[] plaintext(FileChannel ch, long dataStart, long dataLength, byte[] key) throws IOException {
            throw new UnsupportedOperationException("a chunked tensor is decrypted by range, not whole");
        }

        /** Decrypts the chunks overlapping {@code [from, from+length)} into {@code out}. */
        void range(FileChannel ch, long dataStart, long dataLength, long from, byte[] out, int length, byte[] key)
                throws IOException {
            long start = from;
            long end = from + length;
            int chunk = (int) (from / chunkSize);
            while (from < end) {
                long chunkStart = (long) chunk * chunkSize;
                long chunkLength = Math.min(chunkSize, dataLength - chunkStart);
                if (chunkLength <= 0) {
                    throw new IOException("tensorizer chunk " + chunk + " lies outside the tensor");
                }
                if (chunkStart > Integer.MAX_VALUE) {
                    throw new IOException("an encrypted tensor chunk starts past 2^31 bytes");
                }
                byte[] cipher = read(ch, dataStart + chunkStart, (int) chunkLength, Long.MAX_VALUE, null);
                byte[] nonce = increment(initialNonce, chunk);
                TensorizerCrypto.openDetached(key, nonce, macs.get(chunk), cipher);
                long copyFrom = Math.max(from, chunkStart);
                long copyTo = Math.min(end, chunkStart + chunkLength);
                System.arraycopy(cipher, (int) (copyFrom - chunkStart), out, (int) (copyFrom - start),
                        (int) (copyTo - copyFrom));
                from = copyTo;
                chunk++;
            }
        }

        /** The initial nonce incremented {@code by} times as a little-endian 192-bit integer. */
        private static byte[] increment(byte[] initial, int by) {
            byte[] out = initial.clone();
            long carry = by;
            for (int i = 0; i < out.length && carry != 0; i++) {
                long value = (out[i] & 0xffL) + (carry & 0xff);
                out[i] = (byte) value;
                carry = (carry >>> 8) + (value >>> 8);
            }
            return out;
        }
    }

    // ------------------------------------------------------------------ helpers

    private static byte[] key(Path path) throws IOException {
        byte[] key = configuredKey();
        if (key == null) {
            throw new IOException("Invalid tensorizer file '" + path + "': the file is encrypted; set " + KEY_ENV
                    + " (64 hex digits) or " + KEY_FILE_ENV + " to a file holding the 32-byte key");
        }
        return key;
    }

    /** The configured key from the environment, or null. */
    static byte[] configuredKey() throws IOException {
        String hex = System.getenv(KEY_ENV);
        if (hex != null && !hex.trim().isEmpty()) {
            return fromHex(hex.trim());
        }
        String file = System.getenv(KEY_FILE_ENV);
        if (file != null && !file.trim().isEmpty()) {
            byte[] raw = java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(file.trim()));
            if (raw.length == TensorizerCrypto.KEY_BYTES) {
                return raw;
            }
            String text = new String(raw, StandardCharsets.US_ASCII).trim();
            return fromHex(text);
        }
        return null;
    }

    private static byte[] fromHex(String text) throws IOException {
        if (text.length() != TensorizerCrypto.KEY_BYTES * 2) {
            throw new IOException(KEY_ENV + " must hold " + (TensorizerCrypto.KEY_BYTES * 2) + " hex digits, not "
                    + text.length() + " characters");
        }
        byte[] out = new byte[TensorizerCrypto.KEY_BYTES];
        for (int i = 0; i < out.length; i++) {
            int hi = Character.digit(text.charAt(2 * i), 16);
            int lo = Character.digit(text.charAt(2 * i + 1), 16);
            if (hi < 0 || lo < 0) {
                throw new IOException(KEY_ENV + " is not hexadecimal");
            }
            out[i] = (byte) ((hi << 4) | lo);
        }
        return out;
    }

    private static byte[] read(FileChannel ch, long pos, int len, long size, Path path) throws IOException {
        if (pos < 0 || len < 0 || pos > size - len) {
            throw invalid(path, "a read of " + len + " bytes at " + pos + " is outside the file's " + size
                    + " bytes");
        }
        byte[] out = new byte[len];
        ByteBuffer buffer = ByteBuffer.wrap(out);
        long at = pos;
        while (buffer.hasRemaining()) {
            int n = ch.read(buffer, at);
            if (n < 0) {
                throw invalid(path, "the file ended at " + at + " while " + len + " bytes were expected");
            }
            at += n;
        }
        return out;
    }

    private static int byteAt(Path path, byte[] buffer, int offset, String what) throws IOException {
        if (offset < 0 || offset >= buffer.length) {
            throw invalid(path, what + " lies outside its header");
        }
        return buffer[offset] & 0xff;
    }

    private static int uint16(byte[] b, int off) {
        return (b[off] & 0xff) | ((b[off + 1] & 0xff) << 8);
    }

    private static long uint32(byte[] b) {
        return (b[0] & 0xffL) | ((b[1] & 0xffL) << 8) | ((b[2] & 0xffL) << 16) | ((b[3] & 0xffL) << 24);
    }

    private static long uint32(byte[] b, int off) {
        return (b[off] & 0xffL) | ((b[off + 1] & 0xffL) << 8) | ((b[off + 2] & 0xffL) << 16)
                | ((b[off + 3] & 0xffL) << 24);
    }

    private static long uint64(byte[] b, int off) {
        long v = 0;
        for (int i = 7; i >= 0; i--) {
            v = (v << 8) | (b[off + i] & 0xffL);
        }
        return v;
    }

    private static long int64(byte[] b, int off) {
        return uint64(b, off);
    }

    private static IOException invalid(Path path, String reason) {
        return new IOException("Invalid tensorizer file '" + path + "': " + reason);
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }

    @Override
    public String toString() {
        return "TensorizerFile{" + path + ", " + tensors.size() + " tensors}";
    }

    // ------------------------------------------------------------------ tensor data

    /** A tensor's data range, read positionally; encrypted tensors are decrypted per chunk. */
    private static final class TensorizerRowSource implements RowSource {
        private final String name;
        private final FileChannel channel;
        private final long dataStart;
        private final long dataLength;
        private final StDType dtype;
        private final long[] shape;
        private final Crypt crypt;
        private final byte[] key;
        private final ElementFormat format;
        private final long rows;
        private final long rowBytes;

        TensorizerRowSource(String name, FileChannel channel, long dataStart, long dataLength, StDType dtype,
                            long[] shape, Crypt crypt, byte[] key) {
            this.name = name;
            this.channel = channel;
            this.dataStart = dataStart;
            this.dataLength = dataLength;
            this.dtype = dtype;
            this.shape = shape.clone();
            this.crypt = crypt;
            this.key = key;
            this.format = ElementFormat.of(dtype);
            long r = 1;
            for (int i = 0; i < this.shape.length - 1; i++) {
                r = Math.multiplyExact(r, this.shape[i]);
            }
            this.rows = r;
            long innermost = this.shape.length == 0 ? 1 : this.shape[this.shape.length - 1];
            this.rowBytes = format.rowBytes(innermost);
        }

        @Override
        public String describe() {
            return "tensor '" + name + "' (" + dtype.headerName() + " " + Arrays.toString(shape)
                    + ") of the tensorizer file";
        }

        @Override
        public ElementFormat format() {
            return format;
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }

        @Override
        public long rows() {
            return rows;
        }

        @Override
        public long rowBytes() {
            return rowBytes;
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            Objects.requireNonNull(dst, "dst");
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside the " + rows
                        + " rows of " + describe());
            }
            long offset = Math.multiplyExact(first, rowBytes);
            long bytes = Math.multiplyExact(rowBytes, (long) count);
            if (bytes > dst.remaining()) {
                throw new IllegalArgumentException(count + " rows of " + describe() + " need " + bytes
                        + " bytes; the buffer has " + dst.remaining());
            }
            if (bytes == 0) {
                return;
            }
            ByteBuffer window = dst.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            window.limit(window.position() + (int) bytes);
            if (crypt == null) {
                readInto(channel, dataStart + offset, window);
            } else if (crypt instanceof ChunkedCrypt) {
                byte[] plain = new byte[(int) bytes];
                ((ChunkedCrypt) crypt).range(channel, dataStart, dataLength, offset, plain, (int) bytes, key);
                window.put(plain);
            } else {
                byte[] plain = crypt.plaintext(channel, dataStart, dataLength, key);
                if (offset > plain.length - bytes) {
                    throw new IOException(describe() + ": its decrypted bytes do not cover [" + offset + ", +"
                            + bytes + ")");
                }
                window.put(plain, (int) offset, (int) bytes);
            }
            dst.position(dst.position() + (int) bytes);
        }

        private static void readInto(FileChannel ch, long pos, ByteBuffer dst) throws IOException {
            long at = pos;
            while (dst.hasRemaining()) {
                int n = ch.read(dst, at);
                if (n < 0) {
                    throw new IOException("a tensorizer file ended at " + at + " before " + dst + " was filled");
                }
                at += n;
            }
        }

        @Override
        public String toString() {
            return "TensorizerRowSource{" + describe() + "}";
        }
    }
}
