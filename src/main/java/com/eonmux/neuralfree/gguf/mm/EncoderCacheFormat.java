package com.eonmux.neuralfree.gguf.mm;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;

/**
 * The binary form of one {@link EncoderCacheEntry}, as {@link DirectoryEncoderCacheConnector} stores it and as a
 * shared filesystem carries it between servers: the item's key handle, then its chunks in prompt order with their
 * rows, grids, lead pads, canvas tiles and retention masks.
 *
 * <p>The file opens with the magic {@code NFEC} and the format version, holds the payload, and ends with the
 * payload's CRC-32, so a truncated or foreign file is refused by name instead of feeding the prompt garbage. Every
 * count a file declares is checked against the bytes left before anything is allocated. Package-private storage
 * support.
 */
final class EncoderCacheFormat {

    /** The file magic of an encoder cache entry, "NFEC". */
    static final int MAGIC = 0x4E464543;

    /** The format version this build writes and reads. */
    static final int VERSION = 1;

    /** The magic and version ahead of the payload. */
    private static final int HEADER = 8;

    /** The checksum after the payload. */
    private static final int TRAILER = 4;

    private EncoderCacheFormat() {
    }

    /**
     * The file bytes of {@code entry}, stored under {@code keyHandle}.
     */
    static byte[] encode(String keyHandle, EncoderCacheEntry entry) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            out.writeUTF(keyHandle);
            out.writeInt(entry.chunks().size());
            for (EncoderCacheEntry.Chunk chunk : entry.chunks()) {
                writeChunk(out, chunk);
            }
        } catch (IOException e) {
            throw new IllegalStateException("writing an in-memory encoder cache entry failed", e);
        }
        byte[] payload = bytes.toByteArray();
        CRC32 crc = new CRC32();
        crc.update(payload, 0, payload.length);
        ByteBuffer file = ByteBuffer.allocate(HEADER + payload.length + TRAILER);
        file.putInt(MAGIC).putInt(VERSION).put(payload).putInt((int) crc.getValue());
        return file.array();
    }

    /**
     * The entry {@code file} holds, which must have been stored under {@code keyHandle}.
     *
     * @throws IOException when the file is foreign, truncated, corrupt, stored under another key, or holds chunks
     *                     that do not describe one item
     */
    static EncoderCacheEntry decode(String keyHandle, byte[] file) throws IOException {
        if (file.length < HEADER + TRAILER) {
            throw new IOException("the encoder cache entry is " + file.length + " bytes, too short for its header");
        }
        ByteBuffer header = ByteBuffer.wrap(file);
        int magic = header.getInt();
        if (magic != MAGIC) {
            throw new IOException("not a NeuralFree encoder cache entry (magic 0x" + Integer.toHexString(magic) + ")");
        }
        int version = header.getInt();
        if (version != VERSION) {
            throw new IOException("encoder cache entry format " + version + ", this build reads " + VERSION);
        }
        int payloadLength = file.length - HEADER - TRAILER;
        CRC32 crc = new CRC32();
        crc.update(file, HEADER, payloadLength);
        long stored = ByteBuffer.wrap(file, HEADER + payloadLength, TRAILER).getInt() & 0xFFFFFFFFL;
        if (crc.getValue() != stored) {
            throw new IOException("the encoder cache entry is corrupt: checksum 0x" + Long.toHexString(stored)
                    + ", content 0x" + Long.toHexString(crc.getValue()));
        }
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(file, HEADER, payloadLength));
        String storedKey = in.readUTF();
        if (!storedKey.equals(keyHandle)) {
            throw new IOException("the encoder cache entry holds '" + storedKey + "', not '" + keyHandle + "'");
        }
        int chunks = in.readInt();
        if (chunks <= 0 || chunks > payloadLength) {
            throw new IOException("the encoder cache entry declares " + chunks + " chunks in " + payloadLength
                    + " bytes");
        }
        List<EncoderCacheEntry.Chunk> out = new ArrayList<>(chunks);
        for (int i = 0; i < chunks; i++) {
            out.add(readChunk(in, i));
        }
        if (in.available() != 0) {
            throw new IOException("the encoder cache entry has " + in.available() + " bytes after its "
                    + chunks + " chunk(s)");
        }
        try {
            return EncoderCacheEntry.of(out);
        } catch (IllegalArgumentException e) {
            throw new IOException("the encoder cache entry is malformed: " + e.getMessage(), e);
        }
    }

    private static void writeChunk(DataOutputStream out, EncoderCacheEntry.Chunk chunk) throws IOException {
        out.writeByte(chunk.kind().ordinal());
        switch (chunk.kind()) {
            case TEXT:
                out.writeUTF(chunk.text());
                break;
            case IMAGE_ID:
                break;
            case ROWS:
                writeRows(out, chunk.rows());
                break;
            case GRID:
                out.writeInt(chunk.nx());
                out.writeInt(chunk.ny());
                writeRows(out, chunk.rows());
                writeMask(out, chunk.mask());
                break;
            case LEAD_PADDED:
                out.writeInt(chunk.lead());
                writeRows(out, chunk.rows());
                break;
            case CANVAS:
                out.writeInt(chunk.nx());
                out.writeInt(chunk.ny());
                out.writeInt(chunk.canvas().size());
                for (EncoderCacheEntry.CanvasTile tile : chunk.canvas()) {
                    out.writeInt(tile.tokensX());
                    out.writeInt(tile.tokensY());
                    out.writeInt(tile.separators());
                }
                writeRows(out, chunk.rows());
                break;
            default:
                throw new IllegalStateException("unhandled encoder cache chunk " + chunk.kind());
        }
    }

    private static void writeRows(DataOutputStream out, float[][] rows) throws IOException {
        out.writeInt(rows.length);
        out.writeInt(rows[0].length);
        for (float[] row : rows) {
            for (float value : row) {
                out.writeFloat(value);
            }
        }
    }

    private static void writeMask(DataOutputStream out, boolean[] mask) throws IOException {
        if (mask == null) {
            out.writeByte(0);
            return;
        }
        out.writeByte(1);
        for (boolean cell : mask) {
            out.writeByte(cell ? 1 : 0);
        }
    }

    private static EncoderCacheEntry.Chunk readChunk(DataInputStream in, int index) throws IOException {
        int kind = in.readUnsignedByte();
        if (kind >= EncoderCacheEntry.Kind.values().length) {
            throw new IOException("chunk " + index + " has the unknown kind " + kind);
        }
        EncoderCacheEntry.Kind type = EncoderCacheEntry.Kind.values()[kind];
        switch (type) {
            case TEXT:
                return EncoderCacheEntry.Chunk.text(in.readUTF());
            case IMAGE_ID:
                return EncoderCacheEntry.Chunk.imageId();
            case ROWS:
                return EncoderCacheEntry.Chunk.rows(readRows(in, index));
            case GRID: {
                int nx = in.readInt();
                int ny = in.readInt();
                float[][] rows = readRows(in, index);
                return EncoderCacheEntry.Chunk.grid(rows, nx, ny, readMask(in, index, nx, ny));
            }
            case LEAD_PADDED: {
                int lead = in.readInt();
                return EncoderCacheEntry.Chunk.leadPadded(readRows(in, index), lead);
            }
            case CANVAS: {
                int nx = in.readInt();
                int ny = in.readInt();
                int tiles = in.readInt();
                if (tiles <= 0 || tiles > in.available() / 12) {
                    throw new IOException("chunk " + index + " declares " + tiles + " canvas tiles in "
                            + in.available() + " bytes");
                }
                List<EncoderCacheEntry.CanvasTile> canvas = new ArrayList<>(tiles);
                for (int t = 0; t < tiles; t++) {
                    canvas.add(new EncoderCacheEntry.CanvasTile(in.readInt(), in.readInt(), in.readInt()));
                }
                return EncoderCacheEntry.Chunk.canvas(readRows(in, index), nx, ny, canvas);
            }
            default:
                throw new IOException("chunk " + index + " has the unknown kind " + type);
        }
    }

    private static float[][] readRows(DataInputStream in, int index) throws IOException {
        int count = in.readInt();
        int width = in.readInt();
        long values = (long) count * width;
        if (count <= 0 || width <= 0 || values > in.available() / Float.BYTES) {
            throw new IOException("chunk " + index + " describes " + count + "x" + width + " embedding values in "
                    + in.available() + " bytes");
        }
        float[][] rows = new float[count][width];
        for (int r = 0; r < count; r++) {
            for (int c = 0; c < width; c++) {
                rows[r][c] = in.readFloat();
            }
        }
        return rows;
    }

    private static boolean[] readMask(DataInputStream in, int index, int nx, int ny) throws IOException {
        int present = in.readUnsignedByte();
        if (present == 0) {
            return null;
        }
        if (present != 1) {
            throw new IOException("chunk " + index + " has the unknown mask flag " + present);
        }
        long cells;
        try {
            cells = Math.multiplyExact((long) nx, ny);
        } catch (ArithmeticException e) {
            throw new IOException("chunk " + index + " has the impossible grid " + nx + "x" + ny, e);
        }
        if (nx <= 0 || ny <= 0 || cells > in.available()) {
            throw new IOException("chunk " + index + " describes a " + nx + "x" + ny + " grid mask in "
                    + in.available() + " bytes");
        }
        boolean[] mask = new boolean[(int) cells];
        for (int i = 0; i < mask.length; i++) {
            int cell = in.readUnsignedByte();
            if (cell > 1) {
                throw new IOException("chunk " + index + " has the unknown mask value " + cell);
            }
            mask[i] = cell == 1;
        }
        return mask;
    }
}
