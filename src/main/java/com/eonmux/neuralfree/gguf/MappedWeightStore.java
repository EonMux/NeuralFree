package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.utils.LoggingUtil;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Memory-maps the GGUF tensor data region for zero-copy weight access.
 *
 * <p>The OS manages the page cache — hot weights stay in physical RAM transparently.
 * Uses multiple 1GB segments to handle files larger than {@code Integer.MAX_VALUE}.</p>
 *
 * <p>The region's {@link TensorReadMode} decides how tensors are served: {@link TensorReadMode#MMAP} maps them,
 * {@link TensorReadMode#MLOCK} reads them through the channel and pins the file's mapped pages, and
 * {@link TensorReadMode#MMAP_MLOCK} maps and pins them; the remaining modes read through the channel. The mapping
 * takes the paging advice of {@link TensorAdvice}: by default {@code POSIX_MADV_RANDOM} while a NUMA strategy is
 * active, as llama.cpp does on a NUMA host, and otherwise {@code POSIX_MADV_WILLNEED}, llama.cpp's prefetch, for data
 * that fits in RAM.</p>
 *
 * <p>Thread-safe: all reads use absolute-position accessors.</p>
 */
public class MappedWeightStore implements WeightStore {

    /** Maximum bytes per MappedByteBuffer segment (1 GB). */
    private static final long SEGMENT_SIZE = 1024L * 1024 * 1024;

    /** Bytes per mapped segment: {@link #SEGMENT_SIZE}, smaller only where a test needs segment boundaries. */
    private final long segmentSize;
    private final long dataOffset;
    private final long dataLength;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final MappedByteBuffer[] mappings;
    private final boolean mmapAvailable;
    /** Whether {@link #mappings} were pinned with {@code mlock} ({@link TensorReadMode#locksMemory()}). */
    private final boolean pinned;

    // Fallback when mmap fails (32-bit JVM, OS restrictions, etc.)
    private final RandomAccessFile fallbackFile;

    /**
     * Opens the GGUF file and memory-maps the tensor data region.
     *
     * @param path       path to the GGUF file
     * @param dataOffset byte offset where tensor data begins (from {@link GGUFFileParser#getDataOffset()})
     * @param dataLength total bytes of tensor data
     */
    public MappedWeightStore(String path, long dataOffset, long dataLength) throws IOException {
        this(path, dataOffset, dataLength, TensorReadMode.current(), SEGMENT_SIZE);
    }

    /**
     * Opens the file as {@link #MappedWeightStore(String, long, long)} does, under {@code mode} and with mapped
     * segments of {@code segmentSize} bytes.
     */
    MappedWeightStore(String path, long dataOffset, long dataLength, TensorReadMode mode, long segmentSize)
            throws IOException {
        if (segmentSize < 1 || segmentSize > SEGMENT_SIZE) {
            throw new IllegalArgumentException("segment size " + segmentSize + " is not in [1, " + SEGMENT_SIZE + "]");
        }
        this.segmentSize = segmentSize;
        this.dataOffset = dataOffset;
        this.dataLength = dataLength;

        RandomAccessFile raf = null;
        FileChannel ch = null;
        try {
            raf = new RandomAccessFile(path, "r");
            ch = raf.getChannel();

            MappedByteBuffer[] maps = null;
            boolean mapped = false;

            // MLOCK maps the file too, but only to pin its pages: tensor reads stay on the channel
            if (mode.usesMmap() || mode.locksMemory()) {
                try {
                    long fileSize = ch.size();
                    long effectiveDataLen = Math.min(dataLength, fileSize - dataOffset);
                    if (effectiveDataLen <= 0) {
                        throw new IOException("Data region starts beyond EOF: dataOffset="
                                + dataOffset + ", fileSize=" + fileSize);
                    }

                    int segmentCount = (int) ((effectiveDataLen + segmentSize - 1) / segmentSize);
                    if (segmentCount == 0) segmentCount = 1;
                    maps = new MappedByteBuffer[segmentCount];

                    for (int i = 0; i < segmentCount; i++) {
                        long segStart = dataOffset + (long) i * segmentSize;
                        long segLen = Math.min(segmentSize, dataOffset + effectiveDataLen - segStart);
                        if (segLen <= 0) {
                            maps[i] = null;
                            continue;
                        }
                        segLen = Math.min(segLen, fileSize - segStart);
                        if (segLen <= 0) {
                            maps[i] = null;
                            continue;
                        }
                        maps[i] = ch.map(FileChannel.MapMode.READ_ONLY, segStart, segLen);
                    }
                    mapped = true;
                    LoggingUtil.getLogger(MappedWeightStore.class).info(loadModeMessage(mode), mode.flagName(),
                            segmentCount, dataLength / (1024 * 1024));
                } catch (IOException | UnsupportedOperationException e) {
                    maps = null;
                    LoggingUtil.getLogger(MappedWeightStore.class).warn(
                            "MappedWeightStore: {} could not map the data region ({}), using RandomAccessFile"
                                    + " reads", mode.flagName(), e.getMessage());
                }
            } else {
                LoggingUtil.getLogger(MappedWeightStore.class).info(
                        "MappedWeightStore: mmap disabled ({}={}), using RandomAccessFile reads",
                        TensorReadMode.PROP_READ_MODE, mode.flagName());
            }

            this.mmapAvailable = mapped && mode.usesMmap();
            this.mappings = mapped ? maps : null;
            this.pinned = mapped && mode.locksMemory();

            if (pinned) {
                pinMappings(maps);
            }
            final MemoryLock.Advice advice = mapped ? TensorAdvice.forMapping(System.getProperties(), dataLength)
                    : null;
            if (advice != null) {
                for (MappedByteBuffer seg : maps) {
                    if (seg != null) {
                        MemoryLock.advise(seg, advice);
                    }
                }
            }

            // a model larger than RAM only fits beside its host through the mmap page cache; a no-mmap read of it
            // allocates the whole model in heap or swaps
            if (!mmapAvailable) {
                long totalRam = HostMemory.totalBytes();
                if (dataLength > 0 && totalRam > 0 && dataLength > totalRam) {
                    LoggingUtil.getLogger(MappedWeightStore.class).warn(
                            "tensor data of {} MB is larger than RAM of {} MB; without mmap every tensor read"
                                    + " allocates heap beside the model instead of sharing its page cache",
                            dataLength / (1024 * 1024), totalRam / (1024 * 1024));
                }
            }

            this.raf = raf;
            this.channel = ch;
            this.fallbackFile = raf;
        } catch (IOException | RuntimeException e) {
            try { if (ch != null) ch.close(); } catch (IOException ignored) {}
            try { if (raf != null) raf.close(); } catch (IOException ignored) {}
            throw e;
        }
    }

    /** The log line of a mapping created for {@code mode}. */
    private static String loadModeMessage(TensorReadMode mode) {
        if (mode.locksMemory()) {
            return "MappedWeightStore: {} active, {} segment(s), {} MB data region, pinning it in RAM";
        }
        return "MappedWeightStore: {} active, {} segment(s), {} MB data region";
    }

    /** Pins every mapped segment in RAM, reporting how much was actually locked. */
    private static void pinMappings(MappedByteBuffer[] maps) {
        long locked = 0;
        long bytes = 0;
        for (MappedByteBuffer seg : maps) {
            if (seg == null) {
                continue;
            }
            bytes += seg.capacity();
            if (MemoryLock.lock(seg)) {
                locked += seg.capacity();
            }
        }
        if (locked > 0) {
            LoggingUtil.getLogger(MappedWeightStore.class).info(
                    "MappedWeightStore: pinned {} of {} MB of tensor data in RAM",
                    locked / (1024 * 1024), bytes / (1024 * 1024));
        }
    }

    /**
     * Whether tensor reads are served from the memory mapping (not the channel fallback). A mapping created only to
     * pin the file's pages ({@link TensorReadMode#MLOCK}) does not count.
     */
    public boolean isMmapAvailable() {
        return mmapAvailable;
    }

    /**
     * Reads tensor data into a byte array.
     *
     * @param tensorOffset offset relative to the start of the data region (i.e., {@code GGUFTensorInfo.offset()})
     * @param size         number of bytes to read
     * @param dst          destination array (must have at least {@code size} bytes)
     */
    public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
        readTensor(tensorOffset, size, dst, 0);
    }

    /**
     * Reads tensor data into a byte array at a given offset.
     *
     * @param tensorOffset offset relative to the start of the data region
     * @param size         number of bytes to read
     * @param dst          destination array
     * @param dstOffset    offset into dst
     */
    public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
        if (mmapAvailable) {
            readFromMmap(tensorOffset, size, dst, dstOffset);
        } else {
            readFromFile(tensorOffset, size, dst, dstOffset);
        }
    }

    /**
     * Reads tensor data into a direct ByteBuffer (zero-copy path for GPU staging).
     *
     * <p>When mmap is available, this copies from the mapped region to the direct buffer.
     * The OS page cache ensures frequently-accessed weights are served from RAM.</p>
     *
     * @param tensorOffset offset relative to the start of the data region
     * @param size         number of bytes to read
     * @param dst          destination direct ByteBuffer (position will be advanced by {@code size})
     */
    private static final int FALLBACK_CHUNK_SIZE = 1024 * 1024; // 1 MB chunks for fallback

    public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
        if (mmapAvailable) {
            readFromMmapDirect(tensorOffset, size, dst);
        } else {
            // Fallback: read in chunks to avoid large heap allocation
            readFromFileDirect(tensorOffset, size, dst);
        }
    }

    /**
     * A read-only view of the mapped segment holding {@code tensorOffset}, up to {@code maxSize} bytes or the
     * segment's end; null when tensor reads do not come from the mapping ({@link #isMmapAvailable()}).
     */
    @Override
    public ByteBuffer mappedSlice(long tensorOffset, long maxSize) throws IOException {
        if (maxSize < 1) {
            throw new IllegalArgumentException("a mapped slice holds at least one byte, asked for " + maxSize);
        }
        if (!mmapAvailable) {
            return null;
        }
        if (tensorOffset < 0) {
            throw new IOException("negative tensor offset " + tensorOffset);
        }
        long segIdx = tensorOffset / segmentSize;
        int segOff = (int) (tensorOffset % segmentSize);
        MappedByteBuffer seg = segIdx < mappings.length ? mappings[(int) segIdx] : null;
        if (seg == null || segOff >= seg.capacity()) {
            throw new IOException("mapped slice at offset " + (dataOffset + tensorOffset)
                    + " starts beyond the mapped region");
        }
        int length = (int) Math.min(maxSize, seg.capacity() - segOff);
        ByteBuffer view = seg.asReadOnlyBuffer();
        view.position(segOff).limit(segOff + length);
        return view.slice().order(ByteOrder.LITTLE_ENDIAN);
    }

    private void readFromFileDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
        byte[] chunk = new byte[Math.min(size, FALLBACK_CHUNK_SIZE)];
        int remaining = size;
        synchronized (fallbackFile) {
            fallbackFile.seek(dataOffset + tensorOffset);
            while (remaining > 0) {
                int toRead = Math.min(remaining, chunk.length);
                fallbackFile.readFully(chunk, 0, toRead);
                dst.put(chunk, 0, toRead);
                remaining -= toRead;
            }
        }
    }

    private void readFromMmap(long tensorOffset, int size, byte[] dst, int dstOffset)
            throws IOException {
        long pos = tensorOffset; // position within data region
        int remaining = size;
        int dstPos = dstOffset;

        while (remaining > 0) {
            int segIdx = (int) (pos / segmentSize);
            int segOff = (int) (pos % segmentSize);

            if (segIdx >= mappings.length || mappings[segIdx] == null) {
                throw new IOException("Read beyond mapped region: segIdx=" + segIdx
                        + ", offset=" + (dataOffset + pos) + ", remaining=" + remaining);
            }

            MappedByteBuffer seg = mappings[segIdx];
            // Use actual segment capacity — last segment may be shorter than segmentSize
            int readable = (int) Math.min(remaining, seg.capacity() - segOff);
            if (readable <= 0) {
                throw new IOException("Read beyond segment boundary: segOff=" + segOff
                        + ", segCapacity=" + seg.capacity() + ", remaining=" + remaining);
            }

            // Absolute bulk get — thread-safe (no shared position state)
            // Use a duplicate to avoid position conflicts across threads
            ByteBuffer dup = seg.duplicate();
            dup.position(segOff);
            dup.get(dst, dstPos, readable);

            pos += readable;
            dstPos += readable;
            remaining -= readable;
        }
    }

    private void readFromMmapDirect(long tensorOffset, int size, ByteBuffer dst)
            throws IOException {
        long pos = tensorOffset;
        int remaining = size;

        while (remaining > 0) {
            int segIdx = (int) (pos / segmentSize);
            int segOff = (int) (pos % segmentSize);

            if (segIdx >= mappings.length || mappings[segIdx] == null) {
                throw new IOException("Read beyond mapped region: segIdx=" + segIdx
                        + ", offset=" + (dataOffset + pos) + ", remaining=" + remaining);
            }

            MappedByteBuffer seg = mappings[segIdx];
            // Use actual segment capacity — last segment may be shorter than segmentSize
            int readable = (int) Math.min(remaining, seg.capacity() - segOff);
            if (readable <= 0) {
                throw new IOException("Read beyond segment boundary: segOff=" + segOff
                        + ", segCapacity=" + seg.capacity() + ", remaining=" + remaining);
            }

            ByteBuffer dup = seg.duplicate();
            dup.position(segOff);
            dup.limit(segOff + readable);
            dst.put(dup);

            pos += readable;
            remaining -= readable;
        }
    }

    private void readFromFile(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
        synchronized (fallbackFile) {
            fallbackFile.seek(dataOffset + tensorOffset);
            fallbackFile.readFully(dst, dstOffset, size);
        }
    }

    @Override
    public void close() throws IOException {
        // Unmap by nulling references — actual unmap happens at GC
        if (mappings != null) {
            for (int i = 0; i < mappings.length; i++) {
                if (pinned && mappings[i] != null) {
                    MemoryLock.unlock(mappings[i]);
                }
                mappings[i] = null;
            }
        }
        if (channel != null) {
            channel.close();
        }
        // Close the underlying RandomAccessFile to release the file descriptor
        if (raf != null) {
            raf.close();
        }
    }
}
