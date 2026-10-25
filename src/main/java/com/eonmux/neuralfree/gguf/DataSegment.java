package com.eonmux.neuralfree.gguf;

import java.util.Objects;

/**
 * One contiguous data range of a file placed in the virtual tensor address space of a {@link ModelSource}: the
 * {@link #dataLength()} bytes at file offset {@link #fileDataOffset()} hold the virtual offsets
 * {@code [virtualBase, virtualEnd)}. A GGUF split is one segment (its data section); a safetensors file is one segment
 * (the bytes after its header). Immutable.
 */
public final class DataSegment {

    private final String path;
    private final long fileDataOffset;
    private final long virtualBase;
    private final long dataLength;

    /**
     * @param path           the file holding the bytes
     * @param fileDataOffset file offset of the segment's first byte
     * @param virtualBase    virtual offset of the segment's first byte
     * @param dataLength     byte length of the segment; 0 for a file that holds no tensor data
     * @throws IllegalArgumentException when a value is negative or a segment end overflows a signed 64-bit offset
     */
    public DataSegment(String path, long fileDataOffset, long virtualBase, long dataLength) {
        this.path = Objects.requireNonNull(path, "path");
        if (path.isEmpty()) {
            throw new IllegalArgumentException("data segment path is empty");
        }
        if (fileDataOffset < 0 || virtualBase < 0 || dataLength < 0) {
            throw new IllegalArgumentException("data segment of '" + path + "' has a negative value: file offset "
                    + fileDataOffset + ", virtual base " + virtualBase + ", length " + dataLength);
        }
        if (dataLength > Long.MAX_VALUE - fileDataOffset || dataLength > Long.MAX_VALUE - virtualBase) {
            throw new IllegalArgumentException("data segment of '" + path + "' (file offset " + fileDataOffset
                    + ", virtual base " + virtualBase + ", length " + dataLength + ") overflows a 64-bit offset");
        }
        this.fileDataOffset = fileDataOffset;
        this.virtualBase = virtualBase;
        this.dataLength = dataLength;
    }

    /** The file holding the bytes. */
    public String path() { return path; }

    /** File offset of the segment's first byte. */
    public long fileDataOffset() { return fileDataOffset; }

    /** Virtual offset of the segment's first byte. */
    public long virtualBase() { return virtualBase; }

    /** Byte length of the segment. */
    public long dataLength() { return dataLength; }

    /** Virtual offset one past the segment's last byte. */
    public long virtualEnd() {
        return Math.addExact(virtualBase, dataLength);
    }

    /** Whether the virtual range {@code [virtualOffset, virtualOffset + length)} lies inside this segment. */
    public boolean contains(long virtualOffset, long length) {
        return virtualOffset >= virtualBase && length >= 0 && virtualOffset - virtualBase <= dataLength - length;
    }

    /** File offset of the byte at {@code virtualOffset}, which must lie in this segment. */
    public long fileOffset(long virtualOffset) {
        return fileDataOffset + (virtualOffset - virtualBase);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DataSegment)) {
            return false;
        }
        DataSegment other = (DataSegment) o;
        return path.equals(other.path) && fileDataOffset == other.fileDataOffset && virtualBase == other.virtualBase
                && dataLength == other.dataLength;
    }

    @Override
    public int hashCode() {
        int h = path.hashCode();
        h = 31 * h + Long.hashCode(fileDataOffset);
        h = 31 * h + Long.hashCode(virtualBase);
        return 31 * h + Long.hashCode(dataLength);
    }

    @Override
    public String toString() {
        return "DataSegment{" + path + ", data@" + fileDataOffset + ", virtual=[" + virtualBase + ", +" + dataLength
                + ")}";
    }
}
