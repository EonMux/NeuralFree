package com.eonmux.neuralfree.gguf.hf.torch;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.zip.CRC32;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * A {@code torch.save} zip container (a {@code pytorch_model*.bin} / {@code .pt} / {@code .pth} file): its
 * central-directory entries and positional reads of entry bytes, nothing extracted up front. Untrusted input
 * (§3.6): the central directory, the local headers and the deflate streams of a downloaded file are parsed with
 * explicit caps ({@link #MAX_ENTRIES}, {@link #MAX_DIRECTORY_BYTES}, {@link #MAX_INFLATED_BYTES}), the classic and
 * the ZIP64 end-of-central-directory forms are read as torch writes them, and encrypted entries and compression
 * methods beyond stored and deflated are refused with a named error rather than trusted.
 *
 * <p>Container layout (PKWARE APPNOTE as torch's {@code PyTorchStreamWriter} writes it): {@code archive/data.pkl}
 * (the pickle), {@code archive/data/<k>} (one entry per referenced storage) and small {@code archive/} metadata
 * records. Each entry is either stored (method 0, its bytes at {@link Entry#dataOffset()}) or deflated (method 8,
 * read whole against its declared size and CRC-32, with caps). Every read is checked against the entry's declared
 * sizes and CRC-32.</p>
 *
 * <p>Immutable after opening and thread-safe while open; reads are positional and never change the file's bytes.</p>
 */
public final class TorchZipReader implements AutoCloseable {

    /** Most entries of one container (torch makes one {@code archive/data/<k>} entry per storage). */
    public static final int MAX_ENTRIES = 100_000;
    /** Longest entry name accepted. */
    public static final int MAX_ENTRY_NAME = 4096;
    /** Most bytes read for the container's central directory. */
    public static final long MAX_DIRECTORY_BYTES = 256L << 20;
    /** Longest entry read into memory (the pickle, a deflated storage). Mapped storages are read positionally. */
    public static final long MAX_INFLATED_BYTES = 1L << 30;
    /** Longest record tail (a comment plus the ZIP64 locator and record) looked past for the end record. */
    private static final int MAX_TAIL_BYTES = 65_535 + 22 + 20 + 56;

    private static final int END_RECORD = 0x06054b50;
    private static final int END_RECORD_BYTES = 22;
    private static final int ZIP64_END_LOCATOR = 0x07064b50;
    private static final int ZIP64_END_LOCATOR_BYTES = 20;
    private static final int ZIP64_END_RECORD = 0x06064b50;
    private static final int ZIP64_END_RECORD_BYTES = 56;
    private static final int CENTRAL_RECORD = 0x02014b50;
    private static final int CENTRAL_RECORD_FIXED = 46;
    private static final int LOCAL_RECORD = 0x04034b50;
    private static final int LOCAL_RECORD_FIXED = 30;
    private static final int METHOD_STORED = 0;
    private static final int METHOD_DEFLATED = 8;
    private static final int FLAG_ENCRYPTED = 0x0001;
    private static final int ZIP64_EXTRA = 0x0001;
    private static final long DWORD = 0xFFFFFFFFL;

    /** One central-directory record of the container. Immutable. */
    public static final class Entry {

        private final String name;
        private final boolean directory;
        private final boolean stored;
        private final long crc;
        private final long compressedSize;
        private final long size;
        private final long dataOffset;

        Entry(String name, boolean directory, boolean stored, long crc, long compressedSize, long size,
              long dataOffset) {
            this.name = name;
            this.directory = directory;
            this.stored = stored;
            this.crc = crc;
            this.compressedSize = compressedSize;
            this.size = size;
            this.dataOffset = dataOffset;
        }

        /** The entry name, for example {@code archive/data.pkl}. */
        public String name() {
            return name;
        }

        /** True when the name ends in {@code /} (a torch checkpoint's archive records are files). */
        public boolean directory() {
            return directory;
        }

        /** True when the bytes are stored as is (method 0); false for a deflated entry (method 8). */
        public boolean stored() {
            return stored;
        }

        /** The declared CRC-32 of the entry's bytes. */
        public long crc() {
            return crc;
        }

        /** The declared compressed byte length. */
        public long compressedSize() {
            return compressedSize;
        }

        /** The declared uncompressed byte length. */
        public long size() {
            return size;
        }

        /** The absolute file offset of the entry's first byte; -1 for a directory record. */
        public long dataOffset() {
            return dataOffset;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Entry && name.equals(((Entry) o).name);
        }

        @Override
        public int hashCode() {
            return name.hashCode();
        }

        @Override
        public String toString() {
            return "Entry{" + name + ", " + (stored ? "stored" : "deflated") + ", " + size
                    + (stored ? "" : "/" + compressedSize) + " bytes}";
        }
    }

    private final Path path;
    private final FileChannel channel;
    private final List<Entry> entries;
    private final Map<String, Entry> byName;
    private volatile boolean closed;

    private TorchZipReader(Path path, FileChannel channel, List<Entry> entries, Map<String, Entry> byName) {
        this.path = path;
        this.channel = channel;
        this.entries = entries;
        this.byName = byName;
    }

    /**
     * Opens {@code path} for reading and parses its central directory.
     *
     * @throws IOException when the file cannot be opened or is not a torch zip container (prefix {@code "Invalid
     *                     PyTorch checkpoint file '<path>': "})
     */
    public static TorchZipReader open(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        FileChannel ch = FileChannel.open(path, StandardOpenOption.READ);
        try {
            List<Entry> entries = readCentralDirectory(ch, path.toString());
            Map<String, Entry> byName = new LinkedHashMap<>(entries.size());
            for (Entry e : entries) {
                Entry prev = byName.put(e.name, e);
                if (prev != null) {
                    throw new IOException("Invalid PyTorch checkpoint file '" + path + "': entry '" + e.name
                            + "' appears twice in the central directory");
                }
            }
            return new TorchZipReader(path, ch, entries, byName);
        } catch (IOException | RuntimeException e) {
            try {
                ch.close();
            } catch (IOException c) {
                e.addSuppressed(c);
            }
            throw e;
        }
    }

    // ------------------------------------------------------------------ central directory

    /**
     * Locates the end record (classic or ZIP64), checks the declared values, then reads the central directory and
     * resolves every file entry's data offset through its local header.
     */
    private static List<Entry> readCentralDirectory(FileChannel ch, String file) throws IOException {
        long fileBytes = ch.size();
        // The tail of the file holds the end record, optionally a ZIP64 locator before it and a comment after it.
        // The record start is found by scanning the read tail backwards for the signature.
        int tailLen = (int) Math.min(fileBytes, MAX_TAIL_BYTES);
        byte[] tail = new byte[tailLen];
        readFully(ch, ByteBuffer.wrap(tail), fileBytes - tailLen);
        int endAt = -1;
        for (int i = tailLen - END_RECORD_BYTES; i >= 0; i--) {
            if (intAt(tail, i) == END_RECORD) {
                endAt = i;
                break;
            }
        }
        if (endAt < 0) {
            throw invalid(file, "not a torch.save zip container (no end-of-central-directory record)");
        }
        long endOffset = fileBytes - tailLen + endAt;
        long count;
        long dirOffset;
        long dirSize;
        if (endAt >= ZIP64_END_LOCATOR_BYTES && intAt(tail, endAt - ZIP64_END_LOCATOR_BYTES)
                == ZIP64_END_LOCATOR) {
            int locatorAt = endAt - ZIP64_END_LOCATOR_BYTES;
            long recordOffset = longAt(tail, locatorAt + 8);
            if (recordOffset < 0 || recordOffset > endOffset - ZIP64_END_RECORD_BYTES) {
                throw invalid(file, "the ZIP64 end record at " + recordOffset + " does not precede the end record"
                        + " at " + endOffset);
            }
            byte[] rec = new byte[ZIP64_END_RECORD_BYTES];
            readFully(ch, ByteBuffer.wrap(rec), recordOffset);
            if (intAt(rec, 0) != ZIP64_END_RECORD) {
                throw invalid(file, "no ZIP64 end record at its locator's offset " + recordOffset);
            }
            if (intAt(rec, 16) != 0 || intAt(rec, 20) != 0) {
                throw invalid(file, "the ZIP64 end record names disk " + intAt(rec, 16) + "; torch archives"
                        + " have one disk");
            }
            count = longAt(rec, 32);
            dirSize = longAt(rec, 40);
            dirOffset = longAt(rec, 48);
        } else {
            count = shortAt(tail, endAt + 10) & 0xFFFF;
            dirSize = intAt(tail, endAt + 12) & DWORD;
            dirOffset = intAt(tail, endAt + 16) & DWORD;
            if (dirOffset == DWORD || dirSize == DWORD) {
                throw invalid(file, "the central directory is at the ZIP64 boundary but no ZIP64 record precedes"
                        + " the end record (a truncated archive)");
            }
        }
        if (count > MAX_ENTRIES) {
            throw invalid(file, "the container declares " + count + " entries, more than the limit of "
                    + MAX_ENTRIES);
        }
        if (dirSize > MAX_DIRECTORY_BYTES) {
            throw invalid(file, "the central directory is " + dirSize + " bytes, more than the limit of "
                    + MAX_DIRECTORY_BYTES);
        }
        if (dirOffset < 0 || dirOffset > endOffset - dirSize) {
            throw invalid(file, "the central directory at " + dirOffset + " of " + dirSize + " bytes ends past the"
                    + " end record at " + endOffset);
        }
        byte[] dir = new byte[(int) dirSize];
        readFully(ch, ByteBuffer.wrap(dir), dirOffset);
        List<Entry> entries = new ArrayList<>((int) Math.min(count, MAX_ENTRIES));
        int p = 0;
        while (p < dir.length && entries.size() < count) {
            if (dir.length - p < CENTRAL_RECORD_FIXED) {
                throw invalid(file, "truncated central directory at directory byte " + p);
            }
            if (intAt(dir, p) != CENTRAL_RECORD) {
                throw invalid(file, "expected a central directory record at directory byte " + p);
            }
            String name = null;
            boolean directory = false;
            try {
                int flags = shortAt(dir, p + 8) & 0xFFFF;
                int method = shortAt(dir, p + 10) & 0xFFFF;
                long crc = intAt(dir, p + 16) & DWORD;
                long compressedSize = intAt(dir, p + 20) & DWORD;
                long size = intAt(dir, p + 24) & DWORD;
                int nameLen = shortAt(dir, p + 28) & 0xFFFF;
                int extraLen = shortAt(dir, p + 30) & 0xFFFF;
                int commentLen = shortAt(dir, p + 32) & 0xFFFF;
                long localOffset = intAt(dir, p + 42) & DWORD;
                if (dir.length - p < CENTRAL_RECORD_FIXED + nameLen + extraLen + commentLen) {
                    throw invalid(file, "truncated central directory record at directory byte " + p);
                }
                byte[] nameBytes = Arrays.copyOfRange(dir, p + CENTRAL_RECORD_FIXED,
                        p + CENTRAL_RECORD_FIXED + nameLen);
                for (byte b : nameBytes) {
                    if (b == 0) {
                        throw invalid(file, "an entry name holds a NUL");
                    }
                }
                name = new String(nameBytes, StandardCharsets.UTF_8);
                if (name.getBytes(StandardCharsets.UTF_8).length != nameLen) {
                    throw invalid(file, "entry name '" + name + "' is not UTF-8");
                }
                if (nameLen > MAX_ENTRY_NAME) {
                    throw invalid(file, "entry '" + name + "' has a " + nameLen + " byte name, more than the limit"
                            + " of " + MAX_ENTRY_NAME);
                }
                directory = name.endsWith("/");
                if (name.isEmpty() || name.contains("\\")) {
                    throw invalid(file, "entry name '" + name + "' is not a torch archive name");
                }
                if ((flags & FLAG_ENCRYPTED) != 0) {
                    throw invalid(file, "entry '" + name + "' is encrypted");
                }
                boolean cappedSize = size == DWORD;
                boolean cappedCompressed = compressedSize == DWORD;
                boolean cappedOffset = localOffset == DWORD;
                if (cappedSize || cappedCompressed || cappedOffset) {
                    if (method != METHOD_STORED && method != METHOD_DEFLATED) {
                        throw invalid(file, "entry '" + name + "' uses compression method " + method
                                + "; torch archives are stored or deflated");
                    }
                    long[] z64 = zip64Extra(dir, p + CENTRAL_RECORD_FIXED + nameLen, extraLen, cappedSize,
                            cappedCompressed, cappedOffset);
                    if (cappedSize) {
                        size = z64[0];
                        if (size < 0) {
                            throw invalid(file, "entry '" + name + "' declares its size capped but names no ZIP64"
                                    + " value for it");
                        }
                    }
                    if (cappedCompressed) {
                        compressedSize = z64[1];
                        if (compressedSize < 0) {
                            throw invalid(file, "entry '" + name + "' declares its compressed size capped but"
                                    + " names no ZIP64 value for it");
                        }
                    }
                    if (cappedOffset) {
                        localOffset = z64[2];
                        if (localOffset < 0) {
                            throw invalid(file, "entry '" + name + "' declares its local header offset capped but"
                                    + " names no ZIP64 value for it");
                        }
                    }
                }
                if (method != METHOD_STORED && method != METHOD_DEFLATED) {
                    throw invalid(file, "entry '" + name + "' uses compression method " + method
                            + "; torch archives are stored or deflated");
                }
                if (method == METHOD_STORED && compressedSize != size) {
                    throw invalid(file, "stored entry '" + name + "' declares " + size + " bytes but a compressed"
                            + " size of " + compressedSize);
                }
                if (size > fileBytes || compressedSize > fileBytes) {
                    throw invalid(file, "entry '" + name + "' declares " + Math.max(size, compressedSize)
                            + " bytes, more than the file's " + fileBytes);
                }
                long dataOffset = directory ? -1 : dataOffset(ch, file, localOffset, nameBytes);
                if (method == METHOD_STORED && !directory
                        && dataOffset + size > fileBytes) {
                    throw invalid(file, "stored entry '" + name + "' of " + size + " bytes at " + dataOffset
                            + " ends past the file's " + fileBytes + " bytes");
                }
                if (!directory) {
                    entries.add(new Entry(name, directory, method == METHOD_STORED, crc, compressedSize, size,
                            dataOffset));
                }
            } catch (ArrayIndexOutOfBoundsException e) {
                throw invalid(file, "truncated central directory record for entry '" + name + "'", e);
            }
            int nameLen = shortAt(dir, p + 28) & 0xFFFF;
            int extraLen = shortAt(dir, p + 30) & 0xFFFF;
            int commentLen = shortAt(dir, p + 32) & 0xFFFF;
            p += CENTRAL_RECORD_FIXED + nameLen + extraLen + commentLen;
        }
        if (entries.size() != count) {
            throw invalid(file, "the central directory ends after " + entries.size() + " file entries, declaring "
                    + count);
        }
        return entries;
    }

    /**
     * The ZIP64 extra field of a central-directory record: header id {@code 0x0001}, a 2-byte byte count, then the
     * overflowing values in the record's fixed order (uncompressed size, compressed size, local header offset).
     * Absent values stay -1.
     */
    private static long[] zip64Extra(byte[] dir, int extraStart, int extraLen, boolean needSize,
                                     boolean needCompressed, boolean needOffset) throws IOException {
        long[] out = {-1, -1, -1};
        int p = extraStart;
        int end = extraStart + extraLen;
        while (p + 4 <= end) {
            int id = shortAt(dir, p) & 0xFFFF;
            int fieldLen = shortAt(dir, p + 2) & 0xFFFF;
            if (fieldLen < 0 || p + 4 + fieldLen > end) {
                throw new IOException("Invalid PyTorch checkpoint: a central-directory extra field of length "
                        + fieldLen + " ends past its record");
            }
            if (id == ZIP64_EXTRA) {
                int q = p + 4;
                if (needSize) {
                    if (fieldLen < 8) {
                        return out;
                    }
                    out[0] = longAt(dir, q);
                    q += 8;
                    needSize = false;
                }
                if (needCompressed && q + 8 <= p + 4 + fieldLen) {
                    out[1] = longAt(dir, q);
                    q += 8;
                    needCompressed = false;
                }
                if (needOffset && q + 8 <= p + 4 + fieldLen) {
                    out[2] = longAt(dir, q);
                }
                return out;
            }
            p += 4 + fieldLen;
        }
        return out;
    }

    /** The file offset of an entry's first byte, past its local header, whose declared name must match. */
    private static long dataOffset(FileChannel ch, String file, long localOffset, byte[] nameBytes)
            throws IOException {
        String name = new String(nameBytes, StandardCharsets.UTF_8);
        if (localOffset > ch.size() - LOCAL_RECORD_FIXED) {
            throw invalid(file, "entry '" + name + "' has its local header at " + localOffset + ", past the file's "
                    + ch.size() + " bytes");
        }
        ByteBuffer header = ByteBuffer.allocate(LOCAL_RECORD_FIXED);
        readFully(ch, header, localOffset);
        if (header.getInt(0) != LOCAL_RECORD) {
            throw invalid(file, "entry '" + name + "' has no local header at " + localOffset);
        }
        int nameLen = shortAt(header.array(), 26) & 0xFFFF;
        int extraLen = shortAt(header.array(), 28) & 0xFFFF;
        byte[] localName = new byte[nameLen];
        readFully(ch, ByteBuffer.wrap(localName), localOffset + LOCAL_RECORD_FIXED);
        if (!Arrays.equals(localName, nameBytes)) {
            throw invalid(file, "entry '" + name + "' is named '"
                    + new String(localName, StandardCharsets.UTF_8) + "' in its local header");
        }
        long offset = Math.addExact(Math.addExact(localOffset, LOCAL_RECORD_FIXED), nameLen + extraLen);
        if (offset > ch.size()) {
            throw invalid(file, "entry '" + name + "' starts at " + offset + ", past the file's " + ch.size()
                    + " bytes");
        }
        return offset;
    }

    // ------------------------------------------------------------------ reads

    /**
     * Reads {@code dst.remaining()} bytes of {@code entry} from its byte {@code relOff} into {@code dst}: a stored
     * entry is read positionally, a deflated entry is decompressed whole from 0 and checked against its declared
     * size and CRC-32. {@code dst}'s position advances by the bytes read.
     *
     * @throws IllegalArgumentException when the range lies outside the entry, or a deflated read is not whole
     * @throws IOException              when the bytes cannot be read or disagree with the entry's declaration
     */
    public void read(Entry entry, long relOff, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(entry, "entry");
        Objects.requireNonNull(dst, "dst");
        if (closed) {
            throw new IllegalStateException("TorchZipReader of '" + path + "' is closed");
        }
        if (entry.directory) {
            throw new IllegalArgumentException("entry '" + entry.name + "' is a directory record");
        }
        if (relOff < 0 || dst.remaining() < 0 || relOff > entry.size - dst.remaining()) {
            throw new IllegalArgumentException("bytes [" + relOff + ", +" + dst.remaining() + ") lie outside entry '"
                    + entry.name + "' of " + entry.size + " bytes");
        }
        if (!entry.stored) {
            if (relOff != 0 || dst.remaining() != entry.size) {
                throw new IllegalArgumentException("deflated entry '" + entry.name + "' can only be read whole");
            }
            dst.put(inflate(entry));
            return;
        }
        readFully(channel, dst, Math.addExact(entry.dataOffset, relOff));
    }

    /** All bytes of {@code entry} in a new little-endian heap buffer, positioned at 0. */
    public ByteBuffer readAll(Entry entry) throws IOException {
        Objects.requireNonNull(entry, "entry");
        if (entry.size > MAX_INFLATED_BYTES) {
            throw new IOException("Invalid PyTorch checkpoint file '" + path + "': entry '" + entry.name + "' has "
                    + entry.size + " bytes, more than the read limit of " + MAX_INFLATED_BYTES);
        }
        ByteBuffer out = ByteBuffer.allocate((int) entry.size);
        read(entry, 0, out);
        out.order(ByteOrder.LITTLE_ENDIAN);
        out.position(0);
        return out;
    }

    /** The whole deflated entry's bytes, checked against the declared size and CRC-32. */
    private byte[] inflate(Entry entry) throws IOException {
        if (entry.size > MAX_INFLATED_BYTES || entry.compressedSize > MAX_INFLATED_BYTES) {
            throw new IOException("Invalid PyTorch checkpoint file '" + path + "': entry '" + entry.name + "' holds "
                    + Math.max(entry.size, entry.compressedSize) + " bytes, more than the read limit of "
                    + MAX_INFLATED_BYTES);
        }
        Inflater inflater = new Inflater(true);
        try {
            byte[] out = new byte[(int) entry.size];
            byte[] in = new byte[(int) entry.compressedSize];
            readFully(channel, ByteBuffer.wrap(in), entry.dataOffset);
            inflater.setInput(in);
            int done = inflater.inflate(out);
            if (!inflater.finished() && !inflater.needsInput()) {
                byte[] sink = new byte[64 * 1024];
                while (!inflater.finished() && !inflater.needsInput()) {
                    done += inflater.inflate(sink);
                }
            }
            if (!inflater.finished() || done != (int) entry.size) {
                throw invalidEntry(entry, "does not decode to its declared " + entry.size + " bytes (got " + done
                        + ")");
            }
            CRC32 crc = new CRC32();
            crc.update(out, 0, done);
            if (crc.getValue() != entry.crc) {
                throw invalidEntry(entry, "has CRC-32 " + Long.toHexString(crc.getValue()) + ", not its declared "
                        + Long.toHexString(entry.crc));
            }
            return out;
        } catch (DataFormatException e) {
            throw invalidEntry(entry, "is not a valid deflate stream (" + e.getMessage() + ")", e);
        } finally {
            inflater.end();
        }
    }

    private static IOException invalidEntry(Entry e, String reason) {
        return new IOException("Invalid PyTorch checkpoint file '" + e.name + "': " + reason);
    }

    private static IOException invalid(String file, String reason) {
        return new IOException("Invalid PyTorch checkpoint file '" + file + "': " + reason);
    }

    private static IOException invalid(String file, String reason, Throwable c) {
        return new IOException("Invalid PyTorch checkpoint file '" + file + "': " + reason, c);
    }

    private static IOException invalidEntry(Entry e, String reason, Throwable c) {
        return new IOException("Invalid PyTorch checkpoint file '" + e.name + "': " + reason, c);
    }

    // ------------------------------------------------------------------ accessors

    /** The container's file entries in central-directory order (unmodifiable). */
    public List<Entry> entries() {
        return entries;
    }

    /** The entry named {@code name}, or null when there is none. */
    public Entry entry(String name) {
        return byName.get(Objects.requireNonNull(name, "name"));
    }

    /** The file's path as opened. */
    public Path path() {
        return path;
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        channel.close();
    }

    @Override
    public String toString() {
        return "TorchZipReader{" + path + ", " + entries.size() + " entries}";
    }

    // ------------------------------------------------------------------ helpers

    private static int shortAt(byte[] b, int at) {
        return (b[at] & 0xFF) | ((b[at + 1] & 0xFF) << 8);
    }

    private static int intAt(byte[] b, int at) {
        return (b[at] & 0xFF) | ((b[at + 1] & 0xFF) << 8) | ((b[at + 2] & 0xFF) << 16) | ((b[at + 3] & 0xFF) << 24);
    }

    private static long longAt(byte[] b, int at) {
        return (intAt(b, at) & DWORD) | ((intAt(b, at + 4) & DWORD) << 32);
    }

    /** Reads exactly {@code dst.remaining()} bytes of {@code dst} at {@code position}; the position advances. */
    private static void readFully(FileChannel ch, ByteBuffer dst, long position) throws IOException {
        int wanted = dst.remaining();
        while (dst.hasRemaining()) {
            long done = ch.read(dst, position + (wanted - dst.remaining()));
            if (done <= 0) {
                throw new EOFException("read " + (wanted - dst.remaining()) + " of " + wanted + " bytes at "
                        + position);
            }
        }
    }
}
