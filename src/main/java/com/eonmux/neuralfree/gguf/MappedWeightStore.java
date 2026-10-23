package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.utils.LoggingUtil;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Memory-maps the GGUF tensor data region for zero-copy weight access.
 *
 * <p>The OS manages the page cache — hot weights stay in physical RAM transparently.
 * Uses multiple 1GB segments to handle files larger than {@code Integer.MAX_VALUE}.</p>
 *
 * <p>Thread-safe: all reads use absolute-position accessors.</p>
 */
public class MappedWeightStore implements WeightStore {

    /** Maximum bytes per MappedByteBuffer segment (1 GB). */
    private static final long SEGMENT_SIZE = 1024L * 1024 * 1024;

    private final long dataOffset;
    private final long dataLength;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final MappedByteBuffer[] mappings;
    private final boolean mmapAvailable;

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
        this.dataOffset = dataOffset;
        this.dataLength = dataLength;

        RandomAccessFile raf = null;
        FileChannel ch = null;
        try {
            raf = new RandomAccessFile(path, "r");
            ch = raf.getChannel();

            MappedByteBuffer[] maps = null;
            boolean mmapOk = false;

            if (TensorReadMode.current() == TensorReadMode.MMAP) {
                try {
                    long fileSize = ch.size();
                    long effectiveDataLen = Math.min(dataLength, fileSize - dataOffset);
                    if (effectiveDataLen <= 0) {
                        throw new IOException("Data region starts beyond EOF: dataOffset="
                                + dataOffset + ", fileSize=" + fileSize);
                    }

                    int segmentCount = (int) ((effectiveDataLen + SEGMENT_SIZE - 1) / SEGMENT_SIZE);
                    if (segmentCount == 0) segmentCount = 1;
                    maps = new MappedByteBuffer[segmentCount];

                    for (int i = 0; i < segmentCount; i++) {
                        long segStart = dataOffset + (long) i * SEGMENT_SIZE;
                        long segLen = Math.min(SEGMENT_SIZE, dataOffset + effectiveDataLen - segStart);
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
                    mmapOk = true;
                    LoggingUtil.getLogger(MappedWeightStore.class).info(
                            "MappedWeightStore: mmap active, {} segment(s), {} MB data region",
                            segmentCount, dataLength / (1024 * 1024));
                } catch (IOException | UnsupportedOperationException e) {
                    maps = null;
                    LoggingUtil.getLogger(MappedWeightStore.class).warn(
                            "MappedWeightStore: mmap failed ({}), using RandomAccessFile fallback",
                            e.getMessage());
                }
            } else {
                LoggingUtil.getLogger(MappedWeightStore.class).info(
                        "MappedWeightStore: mmap disabled ({}=true), using RandomAccessFile reads",
                        TensorReadMode.PROP_DISABLE_MMAP);
            }

            this.mmapAvailable = mmapOk;
            this.mappings = mmapOk ? maps : null;

            if (mmapOk) {
                this.raf = raf;
                this.channel = ch;
                this.fallbackFile = null;
            } else {
                this.raf = raf;
                this.channel = null;
                this.fallbackFile = raf;
            }
        } catch (IOException | RuntimeException e) {
            try { if (ch != null) ch.close(); } catch (IOException ignored) {}
            try { if (raf != null) raf.close(); } catch (IOException ignored) {}
            throw e;
        }
    }

    /** Returns true if mmap is active (not using fallback). */
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
            int segIdx = (int) (pos / SEGMENT_SIZE);
            int segOff = (int) (pos % SEGMENT_SIZE);

            if (segIdx >= mappings.length || mappings[segIdx] == null) {
                throw new IOException("Read beyond mapped region: segIdx=" + segIdx
                        + ", offset=" + (dataOffset + pos) + ", remaining=" + remaining);
            }

            MappedByteBuffer seg = mappings[segIdx];
            // Use actual segment capacity — last segment may be shorter than SEGMENT_SIZE
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
            int segIdx = (int) (pos / SEGMENT_SIZE);
            int segOff = (int) (pos % SEGMENT_SIZE);

            if (segIdx >= mappings.length || mappings[segIdx] == null) {
                throw new IOException("Read beyond mapped region: segIdx=" + segIdx
                        + ", offset=" + (dataOffset + pos) + ", remaining=" + remaining);
            }

            MappedByteBuffer seg = mappings[segIdx];
            // Use actual segment capacity — last segment may be shorter than SEGMENT_SIZE
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
