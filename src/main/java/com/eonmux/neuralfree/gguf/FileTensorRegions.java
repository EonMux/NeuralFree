package com.eonmux.neuralfree.gguf;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Objects;

/**
 * {@link TensorRegions} over the {@link DataSegment}s of a model's files: one {@link FileChannel} per segment, a
 * tensor found by binary search on {@link DataSegment#virtualBase()}, and each region memory-mapped read-only or, in
 * {@link TensorReadMode#READ} (llama.cpp {@code --no-mmap}), copied into a direct buffer with positional reads. A
 * requested range must lie inside one segment and inside its file. Thread-safe.
 */
public final class FileTensorRegions implements TensorRegions {

    private final String description;
    private final TensorReadMode mode;
    private final DataSegment[] segments;
    private final FileChannel[] channels;

    private FileTensorRegions(String description, TensorReadMode mode, DataSegment[] segments,
                              FileChannel[] channels) {
        this.description = description;
        this.mode = mode;
        this.segments = segments;
        this.channels = channels;
    }

    /**
     * Opens every segment's file for reading.
     *
     * @param segments    the segments in increasing, non-overlapping virtual order
     * @param mode        whether regions are memory-mapped or read into direct buffers
     * @param description the model the regions belong to, for messages
     * @throws IllegalArgumentException when there are no segments or they are out of order or overlap
     * @throws IOException              when a file cannot be opened
     */
    public static FileTensorRegions open(List<DataSegment> segments, TensorReadMode mode, String description)
            throws IOException {
        Objects.requireNonNull(segments, "segments");
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(description, "description");
        DataSegment[] segs = segments.toArray(new DataSegment[0]);
        if (segs.length == 0) {
            throw new IllegalArgumentException("no data segments for " + description);
        }
        for (int i = 0; i < segs.length; i++) {
            Objects.requireNonNull(segs[i], "data segment " + i);
            if (i > 0 && segs[i].virtualBase() < segs[i - 1].virtualEnd()) {
                throw new IllegalArgumentException("data segment " + i + " of " + description + " (" + segs[i]
                        + ") starts before the end of segment " + (i - 1) + " (" + segs[i - 1] + ")");
            }
        }
        FileChannel[] channels = new FileChannel[segs.length];
        try {
            for (int s = 0; s < segs.length; s++) {
                channels[s] = FileChannel.open(Paths.get(segs[s].path()), StandardOpenOption.READ);
            }
        } catch (IOException | RuntimeException e) {
            closeAll(channels, e);
            throw e;
        }
        return new FileTensorRegions(description, mode, segs, channels);
    }

    /** Whether regions are memory-mapped or read into direct buffers. */
    public TensorReadMode mode() {
        return mode;
    }

    @Override
    public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
        Objects.requireNonNull(info, "info");
        if (byteOff < 0 || length < 0 || length > Integer.MAX_VALUE || info.offset() < 0
                || byteOff > Long.MAX_VALUE - info.offset() - length) {
            throw new IOException("tensor " + info.name() + ": region [" + byteOff + ", +" + length + ") at virtual"
                    + " offset " + info.offset() + " is not a valid byte range of " + description);
        }
        long start = info.offset() + byteOff;
        int s = segmentAt(start);
        DataSegment seg = segments[s];
        if (!seg.contains(start, length)) {
            throw new IOException("tensor " + info.name() + " bytes [" + start + ", +" + length + ") lie outside"
                    + " every data segment of " + description);
        }
        long filePos = seg.fileOffset(start);
        FileChannel ch = channels[s];
        long fileSize = ch.size();
        if (filePos > fileSize - length) {
            throw new IOException("tensor " + info.name() + " bytes [" + filePos + ", +" + length + ") lie outside"
                    + " its file '" + seg.path() + "' of " + fileSize + " bytes");
        }
        ByteBuffer buf = mode == TensorReadMode.MMAP ? ch.map(FileChannel.MapMode.READ_ONLY, filePos, length)
                : read(ch, info, filePos, (int) length);
        return buf.order(ByteOrder.LITTLE_ENDIAN);
    }

    /** The last segment whose virtual base is at most {@code virtualOffset}; 0 when the offset precedes them all. */
    private int segmentAt(long virtualOffset) {
        int lo = 0;
        int hi = segments.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (segments[mid].virtualBase() <= virtualOffset) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /** Reads {@code length} bytes at {@code start} into a new direct buffer, positioned at 0. */
    private ByteBuffer read(FileChannel ch, GGUFTensorInfo info, long start, int length) throws IOException {
        ByteBuffer buf;
        try {
            buf = ByteBuffer.allocateDirect(length);
        } catch (OutOfMemoryError e) {
            throw new IOException("cannot allocate " + length + " bytes of direct memory for tensor "
                    + info.name() + " with " + TensorReadMode.PROP_DISABLE_MMAP
                    + "=true; raise -XX:MaxDirectMemorySize or use mmap", e);
        }
        while (buf.hasRemaining()) {
            if (ch.read(buf, start + buf.position()) < 0) {
                throw new EOFException("tensor " + info.name() + " ends at byte " + (start + buf.position())
                        + " of " + description + ", before its " + length + " bytes");
            }
        }
        buf.flip();
        return buf;
    }

    @Override
    public String description() {
        return description;
    }

    /** Closes every file; the first failure is thrown with later ones suppressed. */
    @Override
    public void close() throws IOException {
        IOException first = null;
        for (FileChannel ch : channels) {
            try {
                ch.close();
            } catch (IOException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    private static void closeAll(FileChannel[] channels, Exception cause) {
        for (FileChannel ch : channels) {
            if (ch != null) {
                try {
                    ch.close();
                } catch (IOException e) {
                    cause.addSuppressed(e);
                }
            }
        }
    }

    @Override
    public String toString() {
        return description;
    }
}
