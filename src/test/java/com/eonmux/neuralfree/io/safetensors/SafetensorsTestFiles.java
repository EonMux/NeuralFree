package com.eonmux.neuralfree.io.safetensors;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Test files for the safetensors readers: files with the writer's malformation switches, and raw files whose header
 * text and layout a test spells out (unaligned offsets, defects the writer cannot express).
 */
public final class SafetensorsTestFiles {

    private SafetensorsTestFiles() {
    }

    /** Header defects {@link #writeMalformed} can write; each matches a writer malformation switch. */
    public enum Defect { BAD_LENGTH, OVERLAP, GAP, TRAILING_BYTES, DUPLICATE_KEY, BAD_DTYPE }

    /**
     * Writes a file of two F32 tensors {@code a [2]} and {@code b [3]} carrying {@code defect}.
     */
    public static void writeMalformed(Path file, Defect defect) throws IOException {
        SafetensorsWriter.builder()
                .add("a", StDType.F32, new long[]{2}, new byte[8])
                .add("b", StDType.F32, new long[]{3}, new byte[12])
                .malformed(SafetensorsWriter.Malformation.valueOf(defect.name()))
                .build()
                .write(file);
    }

    /** Writes {@code [u64 LE header length][header UTF-8][data]}, with no padding. */
    public static void writeRaw(Path file, String header, byte[] data) throws IOException {
        writeRaw(file, header.getBytes(StandardCharsets.UTF_8), data);
    }

    /** Writes {@code [u64 LE header length][header][data]}, with no padding. */
    public static void writeRaw(Path file, byte[] header, byte[] data) throws IOException {
        writeRaw(file, header.length, header, data);
    }

    /** Writes {@code [u64 LE lengthField][header][data]}: the length field may disagree with the header. */
    public static void writeRaw(Path file, long lengthField, byte[] header, byte[] data) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 + header.length + data.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(lengthField).put(header).put(data);
        Files.write(file, buf.array());
    }

    /** Bytes {@code first, first + 1, ...} (mod 256). */
    public static byte[] ramp(int first, int n) {
        byte[] out = new byte[n];
        for (int i = 0; i < n; i++) {
            out[i] = (byte) (first + i);
        }
        return out;
    }
}
