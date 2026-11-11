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
 * tensor found by binary search on {@link DataSegment#virtualBase()}, and each region in its
 * {@link TensorReadMode}:
 * <ul>
 *   <li>MMAP: memory-mapped read-only (the page cache backs it), each mapping given the paging advice of
 *       {@link TensorAdvice};</li>
 *   <li>READ (llama.cpp {@code --no-mmap}): copied whole into a direct buffer with positional reads;</li>
 *   <li>DIRECT: copied like READ but through sector-aligned channel reads, so nothing is relied on from the
 *       page cache beyond the reads themselves (the JVM form of llama.cpp's direct I/O);</li>
 *   <li>LAZY ({@code --lazy-mode}): read like READ on demand, per call: row-gathered tensors (per-layer
 *       embeddings and their kin) leave their rows unmapped and unread until the gather asks for them, and their
 *       buffers are not held by this class.</li>
 * </ul>
 * A requested range must lie inside one segment and inside its file. Thread-safe.
 */
public final class FileTensorRegions implements TensorRegions {

    private final String description;
    private final TensorReadMode mode;
    private final MemoryLock.Advice advice;
    private final DataSegment[] segments;
    private final FileChannel[] channels;

    private FileTensorRegions(String description, TensorReadMode mode, MemoryLock.Advice advice,
                              DataSegment[] segments, FileChannel[] channels) {
        this.description = description;
        this.mode = mode;
        this.advice = advice;
        this.segments = segments;
        this.channels = channels;
    }

    /**
     * Opens every segment's file for reading.
     *
     * @param segments    the segments in increasing, non-overlapping virtual order
     * @param mode        how regions are served ({@linkplain TensorReadMode MMAP}, READ, DIRECT or LAZY)
     * @param description the model the regions belong to, for messages
     * @throws IllegalArgumentException when there are no segments, they are out of order or overlap, or
     *                                  {@value TensorAdvice#PROP_ADVICE} names no advice
     * @throws IOException              when a file cannot be opened
     */
    public static FileTensorRegions open(List<DataSegment> segments, TensorReadMode mode, String description)
            throws IOException {
        Objects.requireNonNull(segments, "segments");
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(description, "description");
        DataSegment[] segs = segments.toArray(new DataSegment[0]);
        long mappedBytes = 0;
        for (DataSegment seg : segs) {
            mappedBytes += seg == null ? 0 : seg.dataLength();
        }
        MemoryLock.Advice advice = mode.usesMmap() ? TensorAdvice.forMapping(System.getProperties(), mappedBytes)
                : null;
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
        return new FileTensorRegions(description, mode, advice, segs, channels);
    }

    /** Whether regions are memory-mapped or read into direct buffers. */
    public TensorReadMode mode() {
        return mode;
    }

    /** The paging advice every mapped region takes ({@link TensorAdvice}), or null for none or unmapped modes. */
    public MemoryLock.Advice advice() {
        return advice;
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
        ByteBuffer buf;
        switch (mode) {
            case MMAP:
            case MMAP_MLOCK:
                buf = ch.map(FileChannel.MapMode.READ_ONLY, filePos, length);
                if (advice != null) {
                    MemoryLock.advise(buf, advice);
                }
                if (mode.locksMemory()) {
                    MemoryLock.lock(buf);
                }
                break;
            case DIRECT:
                buf = readAligned(ch, info, filePos, (int) length);
                break;
            default:
                // READ, MLOCK and LAZY: rows are fetched as the gather asks for them
                buf = read(ch, info, filePos, (int) length);
                if (mode.locksMemory()) {
                    MemoryLock.lock(buf);
                }
                break;
        }
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
            throw new IOException("cannot allocate " + length + " bytes of direct memory for tensor " + info.name()
                    + " with mapped reads off; raise -XX:MaxDirectMemorySize or go back to mmap", e);
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

    /**
     * The DIRECT-mode read: the channel serves {@code length} bytes at {@code start} through sector-aligned
     * ({@value #SECTOR}-byte) reads from the aligned start, in a new direct buffer whose slice at the file
     * position is returned, positioned at 0.
     */
    private ByteBuffer readAligned(FileChannel ch, GGUFTensorInfo info, long start, int length) throws IOException {
        long alignedStart = start & ~((long) SECTOR - 1);
        // the aligned window may cross the end of the file (the last tensor of a file, a tiny test model);
        // the channel reads only the sectors that exist
        long window = Math.min(SECTOR * ((start - alignedStart + length + SECTOR - 1) / SECTOR),
                ch.size() - alignedStart);
        int alignedLength = (int) window;
        ByteBuffer aligned;
        try {
            aligned = ByteBuffer.allocateDirect(alignedLength);
        } catch (OutOfMemoryError e) {
            throw new IOException("cannot allocate " + alignedLength + " bytes of aligned direct memory for tensor "
                    + info.name() + " in direct mode; raise -XX:MaxDirectMemorySize or go back to mmap", e);
        }
        while (aligned.hasRemaining()) {
            if (ch.read(aligned, alignedStart + aligned.position()) < 0) {
                throw new EOFException("tensor " + info.name() + " ends at byte " + (alignedStart + aligned.position())
                        + " of " + description + ", before its " + alignedLength + " aligned bytes");
            }
        }
        aligned.flip();
        int rel = (int) (start - alignedStart);
        aligned.limit(rel + length);
        aligned.position(rel);
        return aligned.slice();
    }

    /** Disk sectors per direct read, to copy nothing mapped and to leave nothing read but the sectors asked for. */
    private static final int SECTOR = 4096;

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
