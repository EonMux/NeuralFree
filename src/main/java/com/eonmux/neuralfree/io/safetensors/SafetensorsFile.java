package com.eonmux.neuralfree.io.safetensors;

import com.eonmux.neuralfree.gguf.DataSegment;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * An open safetensors file: its validated {@link SafetensorsHeader} and positional reads of tensor bytes. Consumers
 * that map tensor data use {@link #segment(long)} with the GGUF segment readers ({@code FileTensorRegions},
 * {@code MultiShardMappedWeightStore}); this class never maps. Reads are thread-safe; the file's bytes are
 * little-endian.
 */
public final class SafetensorsFile implements AutoCloseable {

    private final Path path;
    private final FileChannel channel;
    private final SafetensorsHeader header;

    private SafetensorsFile(Path path, FileChannel channel, SafetensorsHeader header) {
        this.path = path;
        this.channel = channel;
        this.header = header;
    }

    /**
     * Opens {@code path} for reading (a symbolic link is followed) and validates its header.
     *
     * @throws IOException when the file cannot be opened or its header is invalid (prefix
     *                     {@code "Invalid safetensors file '<path>': "})
     */
    public static SafetensorsFile open(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        FileChannel ch = FileChannel.open(path, StandardOpenOption.READ);
        try {
            return new SafetensorsFile(path, ch, SafetensorsHeader.read(ch, path.toString()));
        } catch (IOException | RuntimeException e) {
            try {
                ch.close();
            } catch (IOException c) {
                e.addSuppressed(c);
            }
            throw e;
        }
    }

    /** The file's path as opened. */
    public Path path() { return path; }

    /** The validated header. */
    public SafetensorsHeader header() { return header; }

    /** Every tensor, in name order. */
    public List<StTensor> tensors() { return header.tensors(); }

    /** The tensor named {@code name}, or null when the file has none. */
    public StTensor tensor(String name) { return header.tensor(name); }

    /** The {@code __metadata__} entries (empty when absent). */
    public Map<String, String> metadata() { return header.metadata(); }

    /** File offset of the data section: {@code 8 + n}. */
    public long dataStart() { return header.dataStart(); }

    /** Byte length of the data section. */
    public long dataLength() { return header.dataLength(); }

    /** The data section placed at {@code virtualBase} of a virtual tensor address space. */
    public DataSegment segment(long virtualBase) {
        return new DataSegment(path.toString(), dataStart(), virtualBase, dataLength());
    }

    /**
     * Reads {@code dst.remaining()} bytes of {@code t} starting {@code relOff} bytes into it, with positional reads;
     * {@code dst}'s position advances by the bytes read and its byte order is unchanged.
     *
     * @throws IllegalArgumentException when {@code t} is not a tensor of this file or the range lies outside it
     * @throws IOException              when the bytes cannot be read
     */
    public void read(StTensor t, long relOff, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(t, "t");
        Objects.requireNonNull(dst, "dst");
        if (!t.equals(header.tensor(t.name()))) {
            throw new IllegalArgumentException("tensor '" + t.name() + "' is not a tensor of '" + path + "'");
        }
        long len = dst.remaining();
        if (relOff < 0 || relOff > t.byteLength() - len) {
            throw new IllegalArgumentException("bytes [" + relOff + ", +" + len + ") lie outside tensor '"
                    + t.name() + "' of " + t.byteLength() + " bytes in '" + path + "'");
        }
        SafetensorsHeader.readFully(channel, dst, dataStart() + t.begin() + relOff, path.toString());
    }

    /**
     * All bytes of {@code t} in a new little-endian heap buffer positioned at 0.
     *
     * @throws IllegalArgumentException when {@code t} is not a tensor of this file or exceeds a Java array
     * @throws IOException              when the bytes cannot be read
     */
    public ByteBuffer readAll(StTensor t) throws IOException {
        Objects.requireNonNull(t, "t");
        if (t.byteLength() > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("tensor '" + t.name() + "' of " + t.byteLength() + " bytes in '"
                    + path + "' does not fit one buffer");
        }
        ByteBuffer buf = ByteBuffer.allocate((int) t.byteLength());
        read(t, 0, buf);
        buf.flip();
        return buf.order(ByteOrder.LITTLE_ENDIAN);
    }

    /** Closes the file; closing again has no effect. */
    @Override
    public void close() throws IOException {
        channel.close();
    }

    @Override
    public String toString() {
        return "SafetensorsFile{" + path + ", " + header.size() + " tensors}";
    }
}
